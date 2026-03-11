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