"""
==============================================================================
文件名称: active_components.py
所属部门: Components (加工车间)
主要功能: 有源增益介质 ( Yb:CALGO 晶体放大器)
辅食解读: 
    这是全厂最核心的“V8发动机”！
    它包含了两个极其硬核的物理过程：
    1. 踩油门 (pump_crystal)：吸收泵浦光，积攒上能级粒子数（储能）。
    2. 放大 (propagate)：让激光脉冲脱掉相位的衣服，利用切片积分法(Z-Slicing)吸收能量暴涨，
       然后再穿回衣服（恢复复振幅）输出。
==============================================================================
"""

import numpy as np
from components.base_component import BaseComponent
from core.pulse import Pulse
from core.dataclasses_def import CrystalParameters, PumpParameters, PhysicalConstants

class BulkCrystal(BaseComponent):
    """
    有源块状晶体组件 (如 Yb:CALGO)
    能够吸收泵浦能量，并对经过的激光脉冲进行放大。
    """
    def __init__(self, name: str, crystal_params: CrystalParameters, pump_params: PumpParameters):
        super().__init__(name)
        self.cry = crystal_params
        self.pump = pump_params
        self.consts = PhysicalConstants()

        # --- 晶体的内部状态 (Memory) ---
        # 记录当前的上能级粒子数密度 [m^-3] (对应你原代码中的 current_N)
        self.N_upper = 0.0

        # --- 预计算物理量 ---
        # 晶体总体积 V = pi * r^2 * d * N (假设泵浦光斑均匀覆盖)
        self.V = np.pi * (self.pump.w_p **2) * self.cry.thickness * self.cry.num_disks
        # 总掺杂粒子数密度
        self.N_total = self.cry.N_total_base * self.cry.doping_at_percent
        # 晶体总长
        self.L_total = self.cry.thickness * self.cry.num_disks

    def pump_crystal(self, dt: float):
        """
        物理过程 A：泵浦储能 (对应你原代码的 pump_process)
        在 dt 时间内吸收泵浦光，增加上能级粒子数。
        """
        # 泵浦光子能量
        E_pump_photon = self.consts.photon_energy(self.pump.lambda_p)
        
        # 泵浦速率 (单位时间内激发的粒子数密度)
        R_pump = (self.pump.P_pump_avg * self.cry.mode_overlap_efficiency) / (E_pump_photon * self.V)

        # 速率方程 Euler 积分: dN = (R_pump - N/tau) * dt
        dN = (R_pump - self.N_upper / self.cry.tau_f) * dt
        self.N_upper += dN

    def propagate(self, pulse: Pulse) -> Pulse:
        """
        物理过程 B：脉冲放大与能量提取 (采用 Z-Slicing 切片积分法，防止数值爆炸)
        """
        num_slices = 10  # 将晶体切成 10 个薄片进行逐级计算
        dz = self.L_total / num_slices
        
        sigma_em = self.cry.sigma_em_sig_grid
        sigma_abs = self.cry.sigma_abs_sig_grid
        photon_energy_c = self.consts.photon_energy(pulse.grid.lambda_c)
        
        # 让脉冲依次穿过这 10 个薄片
        for _ in range(num_slices):
            # 1. 计算当前薄片下的物理增益系数
            N_lower = self.N_total - self.N_upper
            gain_coeff = (self.N_upper * sigma_em) - (N_lower * sigma_abs)
            
            # 2. 计算这一小段的放大倍数
            G_intensity = np.exp(gain_coeff * dz)
            
            # 记录放大前的能量
            E_in = pulse.get_energy()
            
            # 3. 执行物理放大
            pulse.A_f *= np.sqrt(G_intensity)
            pulse.to_time_domain()  # 同步时域，以便准确获取最新能量
            
            # 4. 计算刚才在这个薄片里提取了多少能量
            E_out = pulse.get_energy()
            E_extracted = E_out - E_in
            extracted_photons = E_extracted / photon_energy_c
            
            # 从全局平均粒子数中扣除被提取的部分 (V 依然是总体积)
            self.N_upper -= extracted_photons / self.V
            
            # 【终极物理保护】：防止数值震荡导致粒子数越界
            # 粒子数绝不可能小于0，也绝不可能超过物理掺杂的总浓度！
            self.N_upper = min(self.N_total, max(0.0, self.N_upper))
            
        return pulse
    

class AdvancedBulkCrystal(BaseComponent):
    
    # 1. 机器的初始化（升级版：加入了泵浦油箱和偏振设置！）
    def __init__(self, name, crystal_params, seed_params, pump_params, consts, pump_polarization='pi'):
        super().__init__(name)
        self.cry = crystal_params
        self.seed = seed_params
        self.pump = pump_params      # ✨ 新增：把外面的泵浦参数接进来，变成自己的油箱
        self.consts = consts
        self.pump_pol = pump_polarization # ✨ 新增：记录泵浦光的偏振方向 (通常是 'pi')
        
        # 你的独门秘籍：记录当前的上能级粒子数密度
        self.N_upper = 0.0
        
        # 加载你需要的截面数据 (假设我们放大的是 sigma 偏振)
        self.sigma_abs = self.cry.sigma_abs_sig_grid
        self.sigma_em  = self.cry.sigma_em_sig_grid
        self.wavelengths = self.cry.wavelength_grid

    # 2. 你的核心物理公式（把你在 v5 里面写的那个函数原封不动抄过来）
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
        pass 

    # 3. ⭐️ 见证奇迹的时刻：流水线对接接口 ⭐️
    def propagate(self, pulse: Pulse) -> Pulse:
        """
        当流水线把 Pulse 传到这台机器时，会触发这个函数
        """
        # 第一步：【脱衣服】
        d_omega = pulse.grid.df * 2 * np.pi
        current_spectrum_J = (np.abs(pulse.A_f)**2) * d_omega 
        phase = np.angle(pulse.A_f) 
        
        # 第二步：【长肌肉】(✨ 这里修复了接口对接 ✨)
        # 把剥离出来的能量，以及机器当前记录的粒子数(self.N_upper)，一起扔进你的核心公式里去暴涨
        # 算完之后，用 spectrum_out_J 接住新能量，用 self.N_upper 接住剩下的粒子数
        spectrum_out_J, self.N_upper = self.amplify_single_pass(current_spectrum_J, self.N_upper)
        
        # 第三步：【穿回衣服】
        pulse.A_f = np.sqrt(spectrum_out_J / d_omega) * np.exp(1j * phase)
        pulse.to_time_domain()
        
        return pulse
    
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
    
    def pump_crystal(self, dt: float):
        """流水线统一标准的充能踏板"""
        # 调用你辛辛苦苦写的动态泵浦过程，并更新机器的当前粒子数
        self.N_upper = self.pump_process(time_duration=dt, N_upper_start=self.N_upper)