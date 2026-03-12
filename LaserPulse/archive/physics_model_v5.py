"""
==============================================================================
文件名称: physics_model_v5.py
所属部门: Archive (历史档案馆)
主要功能: 纯标量光谱放大器模型 (旧版核心引擎)
辅食解读: 
    致敬几个月的心血！这是项目早期的“原型机”。
    它极具物理深度，包含了脉冲串动态平衡、光谱整形等硬核逻辑。
    目前它的精华（泵浦方程、单程放大等）已被移植进最新的 `AdvancedBulkCrystal` 中。
    此文件仅作备忘和物理逻辑参考，不再直接参与流水线运行。
==============================================================================
"""


import numpy as np
from scipy.optimize import brentq
from core.dataclasses_def import SeedParameters, CrystalParameters, CavityParameters, PumpParameters, PhysicalConstants

class SpectralAmplifier:
    def __init__(self, 
                 crystal_params: CrystalParameters, 
                 seed_params: SeedParameters, 
                 pump_params: PumpParameters,
                 cavity_params: CavityParameters,
                 lasing_polarization: str = 'sigma',  # 指定激光振荡偏振
                 pump_polarization: str = 'pi'):      # 指定泵浦吸收偏振
        
        self.consts = PhysicalConstants()
        self.cry = crystal_params
        self.seed = seed_params
        self.pump = pump_params
        self.cav = cavity_params

        self.accumulated_B = 0.0        # B积分累加器
        self.wavelengths = self.cry.wavelength_grid  # 光谱网格

        self.lasing_pol = lasing_polarization
        self.pump_pol = pump_polarization
        
        # --- 1. 加载用于【放大】的光谱数据 (Signal) ---
        # 这里的 sigma_abs 和 sigma_em 专指激光波长方向的截面，与泵浦截面区分开来
        if self.lasing_pol == 'sigma':
            self.sigma_abs = self.cry.sigma_abs_sig_grid
            self.sigma_em  = self.cry.sigma_em_sig_grid
        else:
            self.sigma_abs = self.cry.sigma_abs_pi_grid
            self.sigma_em  = self.cry.sigma_em_pi_grid
        
        # --- 2. 初始化计算 ---
        # 种子光谱初始化
        self.seed_spectrum_J = self._initialize_gaussian_spectrum()
        # 饱和通量计算 (Fluence Saturation)
        self.J_sat_array = self._calculate_saturation_fluence()

    def _initialize_gaussian_spectrum(self) -> np.ndarray:
        """初始化种子光的高斯光谱分布"""
        # 优先使用外部加载的光谱
        if len(self.seed.spectrum_grid) > 0:
            profile = self.seed.spectrum_grid.copy()
        else:
            # 备用：生成理想高斯光谱
            center = self.seed.lambda_s
            bw_fwhm = self.seed.bandwidth
            # 确保单位是米
            if bw_fwhm > 1.0: bw_fwhm *= 1e-9
            sigma = bw_fwhm / 2.355
            profile = np.exp(-0.5 * ((self.wavelengths - center) / sigma)**2)

        area = np.sum(profile)
        if area == 0: return np.zeros_like(self.wavelengths)
        
        # 归一化总能量到 E_seed
        normalized_spectrum = (profile / area) * self.seed.E_seed
        return normalized_spectrum

    def _calculate_saturation_fluence(self) -> np.ndarray:
        """计算饱和通量 J_sat(lambda) = hc / (lambda * (sigma_abs + sigma_em))"""
        h = self.consts.h
        c = self.consts.c
        total_sigma = self.sigma_abs + self.sigma_em
        safe_sigma = np.where(total_sigma < 1e-30, 1e-30, total_sigma)
        return (h * c) / (self.wavelengths * safe_sigma)

    # =========================================================================
    #  核心物理过程 (Core Physics)
    # =========================================================================

    def pump_process(self, time_duration: float, N_upper_start: float) -> float:
        """
        模拟泵浦阶段的粒子数积累
        考虑基态耗尽(Ground State Bleaching)的动态泵浦过程

        :param time_duration: 泵浦持续时间 [s]
        :param N_upper_start: 初始上能级粒子密度 [m^-3]
        """
        # 1. 基础物理量准备
        # 晶体有效体积 (用于浓度与粒子数互转)
        V_crystal = self.seed.seed_area * (self.cry.thickness * self.cry.num_disks)
        
        # 泵浦光子输入速率 (Photons per second) = Power / PhotonEnergy
        photon_E = self.consts.photon_energy(self.pump.lambda_p)
        photons_in_per_sec = self.pump.P_pump_avg / photon_E
        
        # 吸收参数
        sigma_abs = self.cry.get_sigma_at(self.pump.lambda_p, type='abs', polarization=self.pump_pol)
        sigma_em  = self.cry.get_sigma_at(self.pump.lambda_p, type='em',  polarization=self.pump_pol)
        L_total = self.cry.thickness * self.cry.num_disks
        N_doping = self.cry.N_doping
        M_pump = self.pump.M_p
        
        # 2. 时间迭代 (Time Stepping)
        # 将泵浦过程切分为小时间步，以捕捉吸收率随粒子数变化的动态过程
        steps = 50 
        dt = time_duration / steps
        
        current_N_density = N_upper_start
        
        for _ in range(steps):
            # A. 计算当前的基态粒子密度 (Ground State Population)
            N_ground = N_doping - current_N_density
            if N_ground < 0: N_ground = 0
            
            # B. 计算瞬时吸收率 (Beer-Lambert Law)：上能级粒子越多 -> 基态越少 -> 吸收越弱
            # 计算净吸收系数
            effective_length = L_total * M_pump
            alpha_net = (sigma_abs * N_ground) - (sigma_em * current_N_density)
            if alpha_net < 0: alpha_net = 0 # 泵浦光被漂白了，不再吸收
            optical_depth = alpha_net * effective_length
            
            # 限制指数范围防止溢出
            if optical_depth > 100: optical_depth = 100 
            absorbance = 1.0 - np.exp(-optical_depth)
            
            # C. 计算这段 dt 时间内被晶体“截获”的总光子数
            # 考虑模场重叠效率 (Overlap Efficiency)
            absorbed_photons = photons_in_per_sec * dt * absorbance * self.cry.mode_overlap_efficiency
            
            # D. 计算自发辐射损耗掉的粒子数
            # Total Decayed = (N_density * Volume) * (dt / tau)
            total_N_particles = current_N_density * V_crystal
            decayed_particles = total_N_particles * (dt / self.cry.tau_f)
            
            # E. 更新粒子数密度
            # Net Change = (Absorbed - Decayed) / Volume
            delta_N_density = (absorbed_photons - decayed_particles) / V_crystal
            
            current_N_density += delta_N_density
            
            # F. 物理边界限制
            if current_N_density > N_doping: current_N_density = N_doping
            if current_N_density < 0: current_N_density = 0.0
            
        return current_N_density

    def amplify_single_pass(self, current_spectrum_J: np.ndarray, N_upper: float) -> tuple[np.ndarray, float]:
        """
        单程放大模型 (针对均匀加宽介质)
        逻辑：
        1. 计算总输入能量通量 (Total Fluence)。
        2. 计算加权平均的发射/吸收截面 (Effective Sigma)。
        3. 使用标量版 Frantz-Nodvik 计算总提取能量。
        4. 根据总提取能量，反推平均剩余反转粒子数。
        5. 用修正后的粒子数重新计算光谱增益形状。
        """
        L = self.cry.thickness * self.cry.num_disks
        area = self.seed.seed_area
        
        # 1. 计算总通量 (Scalar)
        E_in_total = np.sum(current_spectrum_J)
        J_in_total = E_in_total / area
        
        if J_in_total < 1e-20: # 避免除零
            return current_spectrum_J, N_upper

        # 2. 计算光谱加权后的有效截面 (Effective Cross-sections)
        if E_in_total > 1e-20:
            spectral_weights = current_spectrum_J / E_in_total
            sigma_em_eff = np.average(self.sigma_em, weights=spectral_weights)
            sigma_abs_eff = np.average(self.sigma_abs, weights=spectral_weights)
        else:
            # 能量太低时，默认用中心波长的截面
            idx = len(self.wavelengths) // 2
            sigma_em_eff = self.sigma_em[idx]
            sigma_abs_eff = self.sigma_abs[idx]
        
        # 3. 计算有效饱和通量 (Effective Saturation Fluence)
        # J_sat_eff = hc / (lambda * (sigma_em_eff + sigma_abs_eff))
        # 取中心波长计算 photon energy
        h = self.consts.h
        c = self.consts.c
        lambda_center = self.seed.lambda_s
        J_sat_eff = (h * c) / (lambda_center * (sigma_em_eff + sigma_abs_eff))
        
        # 4. 计算小信号增益 (Small Signal Gain) - 对应当前 N_upper
        # G0 = exp( (N2*sig_em - N1*sig_abs) * L )
        N_lower = self.cry.N_doping - N_upper
        g0_coeff = (N_upper * sigma_em_eff) - (N_lower * sigma_abs_eff)
        G0_total = np.exp(g0_coeff * L)
        
        # 5. 使用标量 Frantz-Nodvik 计算总输出通量
        # J_out = J_sat * ln(1 + G0 * (exp(J_in/J_sat) - 1))
        exp_arg = J_in_total / J_sat_eff
        if exp_arg > 100: exp_arg = 100
        
        term_exp = np.exp(exp_arg)
        term_log = 1.0 + G0_total * (term_exp - 1.0)
        J_out_total = J_sat_eff * np.log(term_log)
        
        # 6. 计算放大后的总能量 & 粒子数消耗
        E_out_total = J_out_total * area
        E_extracted = E_out_total - E_in_total
        
        # 反推消耗了多少粒子数 density
        photon_E = (h * c) / lambda_center
        consumed_density = E_extracted / (photon_E * area * L)
        
        N_upper_new = N_upper - consumed_density
        if N_upper_new < 0: N_upper_new = 0.0
        
        # 7. 重构光谱形状 (Spectral Reshaping)
        # 均匀加宽介质中，光谱形状取决于通过晶体时的“平均增益”
        # 我们假设在脉冲通过期间，平均粒子数密度为 N_avg
        N_avg = (N_upper + N_upper_new) / 2.0 
        N_lower_avg = self.cry.N_doping - N_avg
        
        # 计算针对每个波长的增益系数
        gain_profile = (N_avg * self.sigma_em) - (N_lower_avg * self.sigma_abs)
        G_lambda = np.exp(gain_profile * L)
        
        # 输出光谱 = 输入光谱 * 增益谱
        spectrum_out_J = current_spectrum_J * G_lambda
        
        # 归一化能量 (强制能量守恒，修正由于 N_avg 近似带来的微小误差)
        calc_E_out = np.sum(spectrum_out_J)
        if calc_E_out > 0:
            correction_factor = E_out_total / calc_E_out
            spectrum_out_J *= correction_factor
            
        return spectrum_out_J, N_upper_new

    def propagate_round_trip(self, current_spectrum_J: np.ndarray, N_upper: float) -> tuple[np.ndarray, float]:
        """
        单次往返传播：穿过晶体 -> 腔内损耗 -> (可选腔内整形)
        """
        spec = current_spectrum_J.copy()
        n_up = N_upper
        passes = int(self.seed.M_l) # 单程通过次数

        # 1. 放大 (Gain)
        for _ in range(passes):
            spec, n_up = self.amplify_single_pass(spec, n_up)
            
        # 2. 被动损耗 (Loss)
        spec = spec * (1.0 - self.cav.loss_passive)
        
        # 3. 腔内整形 (Intracavity Shaping) - 预留接口（_apply_math_filter）
        if self.cav.loss_shaping_depth > 0:
            spec = self._apply_math_filter(spec, 
                                           self.cav.loss_shaping_center, 
                                           self.cav.loss_shaping_width, 
                                           self.cav.loss_shaping_depth)
            
        return spec, n_up

    # =========================================================================
    #  工具方法 (Helper Methods)
    # =========================================================================

    def apply_seed_preshaping(self, depth: float, width_nm: float, center_wl: float = None) -> np.ndarray:
        """
        [工具] 产生一个经过腔外预整形的种子光谱副本。
        不会永久修改 self.seed_spectrum_J，而是返回一个新的数组供 simulation 使用。
        """
        # if center_wl is None: center_wl = 1040e-9 # 默认对准增益峰
        
        shaped_seed = self.seed_spectrum_J.copy()
        if depth > 0.001:
            shaped_seed = self._apply_math_filter(shaped_seed, center_wl, width_nm, depth)
            
        return shaped_seed

    def _apply_math_filter(self, spectrum, center, width, depth):
        """
        内部使用的通用高斯陷波滤波器
        数学引擎——无物理状态，仅计算高斯陷波滤波器的传输函数，并应用到输入光谱上。
        """
        delta_lambda = self.wavelengths - center
        order = 2  # 设定阶数
        notch_shape = np.exp( -np.log(2) * ((2 * delta_lambda) / width)**order )
        transmission = 1.0 - (depth * notch_shape)
        
        return spectrum * np.clip(transmission, 0.0, 1.0)
    
    def estimate_thermal_lens(self, absorbed_pump_power: float, extraction_efficiency: float) -> float:
        """
        预留接口：估算热透镜焦距 f_th
        :return: 热透镜焦距 [m] 
        """
        # 1. 计算产热功率 P_heat
        # 量子亏损部分 (Quantum Defect)
        eta_qd = 1.0 - (self.pump.lambda_p / self.seed.lambda_s)
        # 注意：未提取的能量也会部分转化为热，这里做简化估算，假设主要来源是量子亏损
        P_heat = absorbed_pump_power * eta_qd 
        
        # 2. 读取热光参数
        # K_c: 热导率, dn_dT: 热光系数
        # Yb:CALGO (a-cut) K ~ 6.9 W/mK, dn/dT ~ -9e-6 /K (负热透镜!)
        w_p = np.sqrt(self.seed.seed_area / np.pi) # 泵浦/信号光斑半径近似
        
        # 3. 简单的薄透镜公式 f = (2 * K * A) / (P_heat * dn/dT) ... 
        # 这里建议直接引用 Loiko 论文中的公式 (Eq. 5 or similar)
        # 重点是计算出屈光度 D = 1/f
        
        # 这是一个占位返回，需要进一步完善 parameter_loader
        return 0.0

    # =========================================================================
    #  高级仿真接口 (Advanced Simulation Interfaces)
    # =========================================================================

    def run_pulse_train_dynamics(self, 
                                 total_pulses: int, 
                                 round_trips: int,
                                 custom_seed_spectrum: np.ndarray = None) -> dict:
        """
        [核心功能] 运行脉冲串建立过程仿真。
        模拟 泵浦 -> 放大 -> 消耗 -> 下一发 的完整动力学。
        
        Args:
            total_pulses: 模拟脉冲总数
            round_trips: 每个脉冲在腔内往返的次数
            custom_seed_spectrum: (可选) 传入经过预整形的种子光谱。如果不传则用默认种子。
            
        Returns:
            dict: 包含 'energy_history', 'inversion_history', 'final_spectrum' 等数据的字典
        """
        # 1. 准备种子
        if custom_seed_spectrum is not None:
            input_seed = custom_seed_spectrum.copy()
        else:
            input_seed = self.seed_spectrum_J.copy()
            
        # 2. 准备时间参数
        dt_pump = 1.0 / self.seed.freq  # 脉冲间隔时间
        
        # 3. 历史记录容器
        history = {
            'pulse_energy': [],      # 每一发脉冲的输出能量
            'n_upper_before': [],    # 每一发放大前的上能级粒子数
            'n_upper_after': [],      # 每一发放大后的剩余粒子数
            'b_integral': []         # 每一发脉冲累积的非线性相移
        }
        
        # 4. 动力学循环
        current_N = 0.0  # 冷启动 (Cold Start)

        total_pulses = int(total_pulses)
        round_trips = int(round_trips)

        for i in range(total_pulses):    # 宏观循环：模拟一发接一发
            # --- A. 泵浦积累 (Pump Buildup) ---
            # 经过 dt 时间的泵浦，粒子数增加
            current_N = self.pump_process(time_duration=dt_pump, N_upper_start=current_N)
            history['n_upper_before'].append(current_N)
            
            # --- B. 脉冲放大 (Pulse Amplification) ---
            pulse_spec = input_seed.copy()
            temp_N = current_N
            
            self.accumulated_B = 0.0

            # 腔内往返 Loop
            for _ in range(round_trips):        # 微观循环：模拟这一发里的放大
                pulse_spec, temp_N = self.propagate_round_trip(pulse_spec, temp_N)
            
            # --- C. 记录结果与状态更新 ---
            E_out = np.sum(pulse_spec)
            history['pulse_energy'].append(E_out)
            history['n_upper_after'].append(temp_N)
            history['b_integral'].append(self.accumulated_B)

            # 关键：剩余粒子数传递给下一个周期
            current_N = temp_N
            
        # 5. 封装返回结果
        result = {
            'energy_trace': np.array(history['pulse_energy']),
            'n_upper_trace': np.array(history['n_upper_before']),
            'b_integral_trace': np.array(history['b_integral']),
            'final_spectrum': pulse_spec,  # 最后一发的光谱
            'wavelengths': self.wavelengths
        }
        return result