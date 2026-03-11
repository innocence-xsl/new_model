# reproduce_fig2b.py
import matplotlib.pyplot as plt
import numpy as np

# 导入我们的工厂零件
from system.parameter_loader import ParameterLoader
from components.active_components import AdvancedBulkCrystal
from components.passive_components import LossyMirror, SpectralFilter
from system.optical_assembly import RegenerativeAmplifierAssembly
from core.dataclasses_def import PhysicalConstants

def run_simulation(with_shaping: bool, num_round_trips=90):
    """
    这是一个自动化的流水线函数！
    你只要告诉它：加不加整形机 (True/False)，跑多少圈。
    它就会自动进货、组装、跑完并返回最终的脉冲。
    """
    loader = ParameterLoader()
    cry_params = loader.get_crystal_params()
    pump_params = loader.get_pump_params()
    cav_params = loader.get_cavity_params()
    seed_pulse, seed_params = loader.get_seed_params()

    # 🌟 核心控制开关：要不要加腔外预整形？
    if with_shaping:
        total_depth = 0.8  # 挖坑深度 80%
        num_bounces = 7    # 7次反射
        
        # 自动推算单次反射深度
        single_bounce_transmission = (1.0 - total_depth) ** (1.0 / num_bounces)
        depth_per_bounce = 1.0 - single_bounce_transmission
        
        pre_shaper = SpectralFilter(
            name="Extra_Cavity_Pre_Shaper",
            center_wl=1040e-9,  # 🌟 恢复到文献的真实位置
            width=15e-9,        # 🌟 恢复到 15nm 宽度
            depth_per_bounce=depth_per_bounce,
            num_bounces=num_bounces
        )
        # 给种子光来一刀！
        seed_pulse = pre_shaper.propagate(seed_pulse)

    # 组装晶体和反射镜
    consts = PhysicalConstants()
    crystal = AdvancedBulkCrystal(
        name="Yb:CALGO_Crystal",
        crystal_params=cry_params,
        seed_params=seed_params,
        pump_params=pump_params,
        consts=consts,
        pump_polarization='pi'
    )
    mirror = LossyMirror(name="Cavity_Loss_Mirror", reflectivity=1.0 - cav_params.loss_passive)
    
    # 充能 1ms
    crystal.pump_crystal(dt=5e-3)

    # 上流水线跑圈
    cavity_components = [crystal, mirror]
    regen_amp = RegenerativeAmplifierAssembly(
        name=f"Amp_{'Shaped' if with_shaping else 'Unshaped'}", 
        cavity_components=cavity_components
    )
    
    final_pulse = regen_amp.simulate(seed_pulse=seed_pulse, num_round_trips=num_round_trips)
    return final_pulse

def main():
    print("🌟 正在复现文献 Wang et al. (2021) Figure 2(b) 🌟\n")

    # 1. 跑第一遍：无整形 (Unshaped)
    print("--- 正在运行 [无整形] 对照组 ---")
    pulse_unshaped = run_simulation(with_shaping=False, num_round_trips=90)
    spec_unshaped = np.abs(pulse_unshaped.A_f)**2

    # 2. 跑第二遍：有整形 (Shaped)
    print("\n--- 正在运行 [有整形] 实验组 ---")
    pulse_shaped = run_simulation(with_shaping=True, num_round_trips=90)
    spec_shaped = np.abs(pulse_shaped.A_f)**2

    # 3. 拿到波长标尺 (nm)
    wavelengths_nm = pulse_shaped.grid.lambda_window * 1e9

# ==========================================
    # 4. 终极挑战：计算“有整形(Shaped)”脉冲的极限脉宽 (FTL)
    # ==========================================
    # 强行把复振幅的相位清零（纯实数），模拟完美的腔外光栅压缩！
    pulse_shaped.A_f = np.abs(pulse_shaped.A_f) + 0j 
    
    # 跑车时空穿梭，回到时域！
    pulse_shaped.to_time_domain()
    
    # 自动计算飞秒级的时间标尺
    points = pulse_shaped.grid.points
    df = pulse_shaped.grid.df
    dt = 1.0 / (points * df)
    time_fs = np.linspace(-points/2, points/2 - 1, points) * dt * 1e15
    
    # 获取时域强度并归一化
    intensity_t = np.abs(pulse_shaped.A_t)**2
    intensity_t_norm = intensity_t / np.max(intensity_t)

    # 自动寻找半高全宽 (FWHM)
    half_max = 0.5
    indices = np.where(intensity_t_norm >= half_max)[0]
    fwhm_fs = time_fs[indices[-1]] - time_fs[indices[0]]
    
    print(f"\n✨ 完美整形光谱的极限脉冲宽度 (FTL FWHM): {fwhm_fs:.1f} fs ✨")

    # ==========================================
    # 开始画图：双屏展示 (光谱对比 + 时域极限)
    # ==========================================
    print("\n📊 正在生成终极汇报图表...")
    
    # 开一个宽12、高5的画板，分为左右两块屏幕
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- 左屏幕：完美复刻文献的光谱对比 ---
    spec_unshaped_norm = spec_unshaped / np.max(spec_unshaped)
    spec_shaped_norm = spec_shaped / np.max(spec_shaped)

    ax1.plot(wavelengths_nm, spec_unshaped_norm, color='black', linewidth=2.5, label='Without shaping')
    ax1.plot(wavelengths_nm, spec_shaped_norm, color='red', linewidth=2.5, label='With shaping')

    ax1.set_title("Comparison of Amplified Spectra", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Wavelength (nm)", fontsize=12)
    ax1.set_ylabel("Normalized Intensity (a.u.)", fontsize=12)
    ax1.set_xlim(1015, 1065)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- 右屏幕：FTL 极限时域脉冲 ---
    ax2.plot(time_fs, intensity_t_norm, color='purple', linewidth=2)
    
    # 涂色并高亮标出 FWHM
    ax2.fill_between(time_fs, intensity_t_norm, alpha=0.3, color='purple')
    ax2.axvspan(time_fs[indices[0]], time_fs[indices[-1]], color='yellow', alpha=0.3, label=f'FTL: {fwhm_fs:.1f} fs')
    
    ax2.set_title("Transform-Limited Pulse (Shaped)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Time (fs)", fontsize=12)
    ax2.set_ylabel("Normalized Intensity", fontsize=12)
    ax2.set_xlim(-300, 300) # 聚焦在中心 600 飞秒
    ax2.legend(fontsize=12, loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 自动排版并展示
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()