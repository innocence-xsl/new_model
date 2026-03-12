"""
==============================================================================
文件名称: pulse.py
所属部门: Core (核心基建部)
主要功能: 激光脉冲数据载体 (Data Carrier)
辅食解读: 
    这是我们流水线上的主角——“超级跑车”（激光脉冲）。
    它自带两个极其重要的背包：
    1. A_t (时域复振幅)：记录脉冲的形状和时间相位（决定脉冲有多宽）。
    2. A_f (频域复振幅)：记录脉冲包含的光谱颜色和光谱相位。
    它自己还掌握了“时空穿梭”的超能力（FFT/IFFT），能在时间与频率之间自由转换。
==============================================================================
"""

import numpy as np
import scipy.fft as fft
from core.grid import Grid

class Pulse:
    """
    激光脉冲类 (Data Carrier)
    负责承载脉冲的复振幅，并在时域和频域之间自由转换。
    """
    def __init__(self, grid: Grid):
        self.grid = grid
        
        # 频域复振幅 (Complex Amplitude): 包含光谱强度和光谱相位
        self.A_f = np.zeros(self.grid.points, dtype=np.complex128)
        
        # 时域复振幅: 包含时域波形和时域相位（如啁啾）
        self.A_t = np.zeros(self.grid.points, dtype=np.complex128)

    def to_time_domain(self):
        """将频域数据 A_f 转换为时域数据 A_t (IFFT)"""
        # 使用 fftshift 处理零频位置，并乘以缩放因子保持能量量纲正确
        self.A_t = fft.fftshift(fft.ifft(fft.ifftshift(self.A_f))) * self.grid.FFT_scale

    def to_freq_domain(self):
        """将时域数据 A_t 转换为频域数据 A_f (FFT)"""
        self.A_f = fft.fftshift(fft.fft(fft.ifftshift(self.A_t))) / self.grid.FFT_scale

    def get_energy(self) -> float:
        """计算脉冲总能量 (在时域进行积分)"""
        intensity_t = np.abs(self.A_t)**2
        return np.sum(intensity_t) * self.grid.dt

    def get_ftl_pulse(self):
        """
        获取当前光谱对应的傅里叶变换极限 (FTL) 脉冲。
        物理逻辑：抹除所有光谱相位（令相位为0），仅保留振幅，然后进行IFFT。
        """
        ftl_pulse = Pulse(self.grid)
        # 仅取当前复振幅的模长（绝对值），这等于把相位强行设为 0
        ftl_pulse.A_f = np.abs(self.A_f) 
        # 转换到时域
        ftl_pulse.to_time_domain()
        return ftl_pulse