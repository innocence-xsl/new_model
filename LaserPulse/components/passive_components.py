"""
==============================================================================
文件名称: passive_components.py
所属部门: Components (加工车间)
主要功能: 无源光学器件 (如损耗反射镜)
辅食解读: 
    这里放的是不会主动提供能量的“辅助机器”。
    比如 LossyMirror（损耗镜），脉冲经过它时，会被精准扣除掉一部分能量
    （模拟腔内的固有损耗或输出耦合）。
==============================================================================
"""

import numpy as np
from components.base_component import BaseComponent
from core.pulse import Pulse

class LossyMirror(BaseComponent):
    """
    反射镜 (仅带来恒定的能量损耗)
    """
    def __init__(self, name="Mirror", reflectivity=0.99):
        super().__init__(name)
        self.reflectivity = reflectivity

    def propagate(self, pulse: Pulse) -> Pulse:
        # 注意物理逻辑：反射率 (reflectivity) 针对的是能量/强度 (Intensity)。
        # 而我们的 Pulse 承载的是复振幅 (Amplitude)。
        # 能量与振幅的平方成正比，所以振幅的缩放因子应该是反射率的平方根。
        amplitude_factor = np.sqrt(self.reflectivity)
        
        # 同步缩放频域和时域的振幅
        pulse.A_f *= amplitude_factor
        pulse.A_t *= amplitude_factor
        
        return pulse
    
class SpectralFilter(BaseComponent):
    """
    腔内/腔外光谱整形滤波器 (高斯陷波滤波器)
    升级版：支持设定单次反射的深度，以及总共反射的次数 (num_bounces)！
    """
    def __init__(self, name="Spectral_Filter", center_wl=1040e-9, width=15e-9, depth_per_bounce=0.05, num_bounces=1):
        super().__init__(name)
        self.center_wl = center_wl
        self.width = width
        self.depth_per_bounce = depth_per_bounce  # ⚠️ 注意：这里变成了“单次反射深度”
        self.num_bounces = num_bounces            # ⚠️ 新增：反射次数（默认为1次）
        self.order = 2  # 高斯阶数

    def propagate(self, pulse: Pulse) -> Pulse:
        wavelengths = pulse.grid.lambda_window
        delta_lambda = wavelengths - self.center_wl
        notch_shape = np.exp( -np.log(2) * ((2 * delta_lambda) / self.width)**self.order )
        
        # 1. 计算单次反射的透射率 (Single Pass Transmission)
        single_bounce_transmission = 1.0 - (self.depth_per_bounce * notch_shape)
        single_bounce_transmission = np.clip(single_bounce_transmission, 0.0, 1.0)
        
        # 2. 🌟 核心物理升级：计算经过 N 次反射后的总透射率
        # 根据物理定律，连续通过 N 次，总透射率就是单次透射率的 N 次方！
        total_transmission = single_bounce_transmission ** self.num_bounces
        
        # 3. 对脉冲复振幅进行整形
        amplitude_factor = np.sqrt(total_transmission)
        pulse.A_f *= amplitude_factor
        
        # 同步时域
        pulse.to_time_domain()
        
        return pulse
    
class AdvancedMaterialDispersion(BaseComponent):
    """
    真实介质色散器
    不仅包含群速度色散(Beta 2)，还加入了让脉冲极度扭曲的高阶色散(Beta 3)！
    """
    def __init__(self, name="Real_Dispersion", length=8e-3, beta2=30e-27, beta3=40e-42):
        super().__init__(name)
        self.length = length
        self.beta2 = beta2 # 群速度色散 s^2/m (会导致拉宽)
        self.beta3 = beta3 # 三阶色散 s^3/m (会导致脉冲前后不对称)

    def propagate(self, pulse: Pulse) -> Pulse:
        delta_omega = pulse.grid.omega_window - pulse.grid.omega_c
        
        # 泰勒展开计算真实相位延迟
        phase = (0.5 * self.beta2 * (delta_omega**2) + 
                 (1/6) * self.beta3 * (delta_omega**3)) * self.length
        
        pulse.A_f *= np.exp(-1j * phase)
        pulse.to_time_domain()
        return pulse

class TreacyGratingCompressor(BaseComponent):
    """
    真实的 Treacy 平行光栅压缩器
    数学模型完美复刻自 pyLaserPulse 的 grating_compressor.get_Taylors()
    """
    def __init__(self, name="Treacy_Compressor", groove_density_mm=1000,
                 input_angle_deg=45, grating_separation=0.1):
        super().__init__(name)
        # 将线数(lines/mm)转换为光栅周期(m)
        self.groove_spacing = 1.0 / (groove_density_mm * 1e3) 
        self.input_angle = np.radians(input_angle_deg)
        self.grating_separation = grating_separation

    def auto_tune_separation(self, target_beta2_to_compensate, omega_c):
        """厂长特权：自动推算到底需要多宽的距离，才能完美抵消晶体的 Beta 2"""
        import scipy.constants as const
        diff_angle = np.arcsin((2 * np.pi * const.c / omega_c) / self.groove_spacing - np.sin(self.input_angle))

        # 这里的 factor_1 和 factor_3 完全取自 pyLaserPulse 原版物理公式
        factor_1 = -8 * np.pi**2 * const.c / (omega_c**3 * self.groove_spacing**2)
        factor_3 = 1 - ((2 * np.pi * const.c / (omega_c * self.groove_spacing)) - np.sin(self.input_angle))**2
        beta2_per_m = factor_1 / (np.cos(diff_angle) * factor_3)

        self.grating_separation = target_beta2_to_compensate / beta2_per_m
        return self.grating_separation

    def propagate(self, pulse: Pulse) -> Pulse:
        import scipy.constants as const
        omega_c = pulse.grid.omega_c
        omega = pulse.grid.omega_window

        diff_angle = np.arcsin((2 * np.pi * const.c / omega_c) / self.groove_spacing - np.sin(self.input_angle))

        # 1. 严格计算光栅的 Beta 2
        factor_1 = -8 * np.pi**2 * const.c / (omega_c**3 * self.groove_spacing**2)
        factor_2 = self.grating_separation / np.cos(diff_angle)
        factor_3 = 1 - ((2 * np.pi * const.c / (omega_c * self.groove_spacing)) - np.sin(self.input_angle))**2
        beta_2 = factor_1 * factor_2 / factor_3

        # 2. 严格计算光栅带来的致命 Beta 3
        factor_1_b3 = -(3 / omega_c)
        factor_2_b3 = (1 + (2 * np.pi * const.c / (omega_c * self.groove_spacing)) * np.sin(self.input_angle) - np.sin(self.input_angle)**2)
        factor_3_b3 = 1 / factor_3
        beta_3 = factor_1_b3 * factor_2_b3 * factor_3_b3 * beta_2

        print(f"   [真实光栅阵列] 自动匹配距离: {self.grating_separation*1000:.2f} mm")
        print(f"   [真实光栅阵列] 提供补偿 Beta 2: {beta_2*1e27:.2f} fs^2")
        print(f"   [真实光栅阵列] 附带致命的 Beta 3: {beta_3*1e42:.2f} fs^3")

        # 3. 施加真实的高阶相位（不完美压缩！）
        delta_omega = omega - omega_c
        phase = 0.5 * beta_2 * (delta_omega**2) + (1/6) * beta_3 * (delta_omega**3)

        pulse.A_f *= np.exp(-1j * phase)
        pulse.to_time_domain()
        return pulse