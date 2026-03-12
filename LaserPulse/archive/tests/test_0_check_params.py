"""
TEST 0: 参数加载与完整性校验
"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from parameter_loader import ParameterLoader

matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans'] # 优先用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题

class ParameterTester:
    def __init__(self):
        print(f"{'='*60}")
        print(" 🧪 TEST 1: 系统参数加载与可视化诊断 (System Diagnosis)")
        print(f"{'='*60}")
        
        try:
            self.loader = ParameterLoader()
            print(f"✅ ParameterLoader 初始化成功 | Base Dir: {self.loader.base_dir}")
        except Exception as e:
            print(f"❌ ParameterLoader 初始化失败: {e}")
            sys.exit(1)

    # --- 1. 纯参数检查部分 (严格保留你的变量名) ---

    def check_pump_params(self):
        print(f"\n[1/4] 检查泵浦源参数 (Pump Parameters)")
        print(f"{'-'*40}")
        pump = self.loader.get_pump_params()
        try:
            print(f"  - 平均功率 (P_pump_avg)     : {pump.P_pump_avg:.1f} W")
            print(f"  - 波长 (lambda_p)           : {pump.lambda_p * 1e9:.1f} nm")
            print(f"  - 光斑半径 (w_p)            : {pump.w_p * 1e6:.1f} um")
            print(f"  - 通光次数 (M_p)            : {pump.M_p} 通")
            print(f"  - 占空比 (duty_cycle)       : {pump.duty_cycle * 100:.1f} %")
        except AttributeError as e:
            print(f"❌ 泵浦参数读取错误: {e}")

    def check_cavity_params(self):
        print(f"\n[2/4] 检查腔体参数 (Cavity Parameters)")
        print(f"{'-'*40}")
        cav = self.loader.get_cavity_params()
        try:
            print(f"  - 腔长 (length)             : {cav.length:.1f} m")
            print(f"  - 被动损耗 (loss_passive)    : {cav.loss_passive*100:.2f} %")
            print(f"  - 腔模半径 (mode_radius)     : {cav.mode_radius*1e6:.1f} um")
            # 物理计算检查
            rt_time = 2 * cav.length / 3e8
            print(f"  > [计算] 单次往返时间 : {rt_time*1e9:.2f} ns")
            print(f"  > [计算] 基础重复频率 : {1/rt_time/1e6:.2f} MHz")
        except AttributeError as e:
             print(f"❌ 腔体参数读取错误: {e}")

    def check_seed_params(self):
        print(f"\n[3/4] 检查种子光参数 (Seed Parameters)")
        print(f"{'-'*40}")
        s = self.loader.get_seed_params(align_to_peak=False)
        try:
            print(f"  - 中心波长 (lambda_s)       : {s.lambda_s*1e9:.1f} nm")
            print(f"  - 脉冲宽度 (bandwidth)      : {s.bandwidth*1e9:.1f} nm")
            print(f"  - 单脉冲能量 (E_seed)       : {s.E_seed*1e9:.2f} nJ")
            print(f"  - 种子光斑半径 (w_s)        : {s.w_s*1e6:.1f} um")
            print(f"  - 重复频率 (freq)           : {s.freq/1e6:.2f} MHz")
            print(f"  - 往返圈数 (round_trips)    : {s.round_trips:.0f} 圈")
        except AttributeError as e:
            print(f"❌ 种子参数读取错误: {e}")

    def check_crystal_params(self):
        print(f"\n[4/4] 检查晶体参数 (Crystal Parameters)")
        print(f"{'-'*40}")
        cry = self.loader.get_crystal_params()
        try:
            print(f"  - 晶体长度 (thickness)       : {cry.thickness*1e3:.2f} mm")
            print(f"  - 晶体数量 (num_disks)       : {cry.num_disks} 个")
            print(f"  - 掺杂浓度 (doping_at_percent): {cry.doping_at_percent:.1f} %")
            print(f"  - 基质离子密度 (N_total_base): {cry.N_total_base:.2e} ")
            print(f"  - 上能级寿命 (tau_f)         : {cry.tau_f * 1e6:.1f} μs ")
        except AttributeError as e:
            print(f"❌ 晶体参数读取错误: {e}")

        # --- 晶体截面绘图 (核心修改：共用纵坐标) ---
        print("\n  >> 正在绘制晶体发射/吸收截面...")
        wl = self.loader.master_wavelengths
        # 统一单位转换 (1e-24 m² -> 1e-20 cm²)
        abs_pi = cry.sigma_abs_pi_grid * 1e24
        em_pi = cry.sigma_em_pi_grid * 1e24
        abs_sig = cry.sigma_abs_sig_grid * 1e24
        em_sig = cry.sigma_em_sig_grid * 1e24

        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # --- 上半部分：π偏振 (共用纵坐标) ---
        c_pi_abs = 'tab:blue'
        c_pi_em = 'tab:red'
        ax1.set_xlabel('Wavelength (nm)')
        # 统一纵坐标标签
        ax1.set_ylabel('Cross-section ($10^{-20} cm^2$)', fontsize=10)
        # 绘制吸收和发射曲线在同一坐标轴
        l1 = ax1.plot(wl*1e9, abs_pi, color=c_pi_abs, label='Abs ($\\pi$)', linewidth=2)
        l2 = ax1.plot(wl*1e9, em_pi, color=c_pi_em, label='Em ($\\pi$)', linewidth=2, linestyle='--')
        # 添加图例
        ax1.legend(loc='upper right', fontsize=9)
        ax1.set_title('π-Polarization Cross-sections', fontweight='bold')
        ax1.grid(alpha=0.3) # 增加网格更易对比

        # --- 下半部分：σ偏振 (共用纵坐标) ---
        c_sig_abs = 'tab:cyan'
        c_sig_em = 'tab:orange'
        ax2.set_xlabel('Wavelength (nm)')
        # 统一纵坐标标签
        ax2.set_ylabel('Cross-section ($10^{-20} cm^2$)', fontsize=10)
        # 绘制吸收和发射曲线在同一坐标轴
        l3 = ax2.plot(wl*1e9, abs_sig, color=c_sig_abs, label='Abs ($\\sigma$)', linewidth=2)
        l4 = ax2.plot(wl*1e9, em_sig, color=c_sig_em, label='Em ($\\sigma$)', linewidth=2, linestyle='--')
        # 添加图例
        ax2.legend(loc='upper right', fontsize=9)
        ax2.set_title('σ-Polarization Cross-sections', fontweight='bold')
        ax2.grid(alpha=0.3) # 增加网格更易对比
        fig.suptitle('Yb:CALGO Dual-Polarization Cross-sections', fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_path = "result_step0_crystal.png"
        plt.savefig(save_path)
        print(f"\n📈 验证曲线已保存至: {save_path}")
        plt.show()

    def run_all(self):
        self.check_pump_params()
        self.check_crystal_params()
        self.check_cavity_params()
        self.check_seed_params()
        
        print(f"\n{'='*60}")
        print("🎉 参数检查结束 (Parameter Check Completed)")
        print(f"{'='*60}")

if __name__ == "__main__":
    tester = ParameterTester()
    tester.run_all()