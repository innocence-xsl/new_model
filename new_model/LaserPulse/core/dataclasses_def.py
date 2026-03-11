"""
==============================================================================
文件名称: dataclasses_def.py
所属部门: Core (核心基建部)
主要功能: 物理参数与设备规格说明书 (Data Classes)
辅食解读: 
    这里是整个工厂的“参数大本营”。
    我们用 Python 的 @dataclass 把零散的变量打包管理起来。
    包含：物理常数(光速/普朗克常数)、晶体参数(Yb:CALGO)、泵浦参数(油箱大小)、
    种子光参数以及谐振腔参数。找参数来这里，一目了然！
==============================================================================
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Optional

# ==========================================
# 1. 物理常数 (Physical Constants)
# ==========================================
@dataclass(frozen=True)  # 设置为不可变，防止误修改
class PhysicalConstants:
    """基础物理常数"""
    c: float = 299792458.0       # 光速 [m/s]
    h: float = 6.62607015e-34    # 普朗克常数 [J·s]
    
    # 辅助计算属性
    def photon_energy(self, wavelength_m: float) -> float:
        """计算单个光子能量 E = hc/lambda"""
        return (self.h * self.c) / wavelength_m

# ==========================================
# 2. Yb:CALGO 晶体参数 (Crystal Parameters)
# ==========================================

@dataclass
class CrystalParameters:
    """
    定义增益介质 Yb:CALGO 的物理属性。
    新增：光谱数组字段，用于飞秒脉冲模拟
    """
    # --- 几何与基础参数 ---
    thickness: float = 0.0          # 晶体长度 [m]
    num_disks: int = 1              # 晶体数量
    doping_at_percent: float = 0.0  # 掺杂浓度 (at.%)
    N_total_base: float = 1.25e28   # 基质离子密度基数 [ions/m^3]
    tau_f: float = 420e-6           # 上能级寿命 [s]
    mode_overlap_efficiency: float = 0.90    # 模式匹配效率
    thermal_loss_coefficient: float = 0.0015 # 热损耗系数
    nyb: float = 1.93               # 折射率
    n2: float = 4.0e-20             # 非线性折射率 [m^2/W]
    K_c:float = 6.9                 # 热导率
    dn_dT:float = 0.0               # 热光系数
    
    # --- 光谱数据 (由 Loader 填充) ---
    # 定义标准波长网格 (Master Grid)
    wavelength_grid: np.ndarray = field(default_factory=lambda: np.array([]))

    # 定义 π 偏振和 σ 偏振的吸收和发射截面
    sigma_abs_pi_grid: np.ndarray = None   # π 偏振吸收
    sigma_em_pi_grid: np.ndarray = None    # π 偏振发射
    sigma_abs_sig_grid: np.ndarray = None # σ 偏振吸收
    sigma_em_sig_grid: np.ndarray = None  # σ 偏振发射

    @property
    def N_doping(self) -> float:
        """计算掺杂离子密度 [ions/m^3]"""
        return self.N_total_base * (self.doping_at_percent / 100)

    def get_sigma_at(self, wavelength_m: float, type: str = 'abs', polarization: str = 'pi') -> float:
        """
        辅助函数：获取特定波长和偏振方向的截面值
        :param type: 'abs' 或 'em'
        :param polarization: 'pi' 或 'sigma' (默认为 'pi'，通常用于泵浦)
        """
        if len(self.wavelength_grid) == 0:
            return 0.0
            
        # 1. 根据请求的偏振和类型选择正确的数据数组
        if polarization.lower() == 'pi':
            arr = self.sigma_abs_pi_grid if type == 'abs' else self.sigma_em_pi_grid
        elif polarization.lower() == 'sigma':
            arr = self.sigma_abs_sig_grid if type == 'abs' else self.sigma_em_sig_grid
        else:
            raise ValueError(f"Unknown polarization: {polarization}")

        if arr is None:
            return 0.0

        # 2. 插值获取数值
        return np.interp(wavelength_m, self.wavelength_grid, arr)


# ==========================================
# 3. 泵浦参数 (Pump Parameters)
# ==========================================
@dataclass
class PumpParameters:
    """
    定义泵浦源及泵浦系统的参数。
    特别针对薄片激光器的多通泵浦设计。
    """
    P_pump_avg: float = 150.0    # 平均泵浦功率 [W]
    lambda_p: float = 980e-9     # 泵浦中心波长 [m]
    
    # 空间参数
    w_p: float = 206e-6          # 泵浦光斑半径 [m] (高斯光束 1/e^2)
    M_p: int = 24                # 泵浦通数 (Pump passes)，薄片常用 24 或 48 通
    
    # 时间参数 (针对脉冲泵浦或 CW 泵浦的模拟)
    is_pulsed: bool = False      # 是否为脉冲泵浦
    rep_rate: float = 1.0        # 泵浦重复频率 [Hz] (如果是 CW，此项无效)
    duty_cycle: float = 1.0      # 占空比 (CW = 1.0)

    # --- 自动计算的属性 ---
    @property
    def pump_area(self) -> float:
        """泵浦光斑面积 [m^2]"""
        return np.pi * self.w_p**2

    @property
    def intensity(self) -> float:
        """泵浦功率密度 [W/m^2]"""
        if self.pump_area > 0:
            return self.P_pump_avg / self.pump_area
        return 0.0

# ==========================================
# 4. 种子/信号光参数 (Seed/Signal Parameters)
# ==========================================
@dataclass
class SeedParameters:
    """
    定义注入再生放大器的种子激光参数。
    """
    lambda_s: float = 1030e-9    # 种子光中心波长 [m]
    E_seed: float = 10e-9        # 单脉冲能量 [J] (注意：F-N计算用能量，不用功率)
    
    # 光谱带宽参数：指定初始带宽
    bandwidth: float = 27.0   # 种子光带宽 (FWHM) [nm]
    spectrum_grid: np.ndarray = field(default_factory=lambda: np.array([]))  # 种子光谱数组 (由 Loader 填充)
    
    # 空间参数 (需与泵浦光斑匹配)
    w_s: float = 200e-6          # 种子光斑半径 [m] (通常略小于泵浦光斑以获得好光束质量)
    
    # 时间参数
    freq: float = 1e6            # 重复频率 [Hz] (1 MHz)
    tau_stretched: float = 500e-12 # 展宽后的脉宽 [s]  | 假设值500ps
    
    # 腔内参数
    M_l: int = 1                 # 单次往返通过晶体的次数 (驻波腔=2, 环形腔=1, 薄片V腔=2等)
    round_trips: int = 50        # 计划放大的往返圈数

    # 腔外光谱整形参数，用于抑制增益窄化，实现百飞秒脉宽
    shaping_depth: float = 0.6          # 挖坑深度 [0.0 - 1.0], 0表示不开启
    shaping_center: float = 1040e-9     # 滤波中心波长 [m]
    shaping_width: float = 20e-9      # 滤波带宽 (FWHM) [nm] - 为了直观常用nm，在代码里乘e-9

    @property
    def seed_area(self) -> float:
        """种子光斑面积 [m^2]"""
        return np.pi * self.w_s**2

    @property
    def input_fluence(self) -> float:
        """种子输入通量 [J/m^2]"""
        if self.seed_area > 0:
            return self.E_seed / self.seed_area
        return 0.0

# ==========================================
# 5. 腔参数 (Cavity Parameters)
# ==========================================
# dataclasses_def.py (部分代码)

@dataclass
class CavityParameters:
    """
    定义再生放大器的谐振腔物理属性。
    对应 parameters/cavity.txt
    """
    # --- 基础参数 ---
    length: float = 1.5          # 腔长 [m]
    loss_passive: float = 0.02   # 被动损耗
    loss_active: float = 0.0     # 主动损耗
    arm_ratio: float = 0.6
    extraction_efficiency: float = 0.9 # 提取效率

    # --- 腔内整形参数 ---
    loss_shaping_depth: float = 0.05         # 挖坑深度 [0.0 - 1.0], 0表示不开启
    loss_shaping_center: float = 1040e-9     # 滤波中心波长 [m]
    loss_shaping_width: float = 15e-9      # 滤波带宽 (FWHM) [nm] - 为了直观常用nm，在代码里乘e-9

    # --- 模场与光束质量 ---
    mode_radius: float = 200e-6  # 腔模半径 [m]
    M2_factor: float = 1.0       # 光束质量
    
    # --- 几何设计 ---
    roc_concave: float = 0.0     # 曲率半径
    angle_fold: float = 0.0      # 折叠角

    @property
    def transmission(self) -> float:
        """单程传输率 T = 1 - Loss"""
        return 1.0 - (self.loss_passive + self.loss_active)
    
    @property
    def round_trip_time(self) -> float:
        """往返时间 [s]"""
        # Trt = 2 * L / c
        return 2 * self.length / PhysicalConstants.c
    
    @property
    def round_trip_loss(self) -> float:
        """往返总损耗 (方便调用)"""
        return self.loss_passive + self.loss_active