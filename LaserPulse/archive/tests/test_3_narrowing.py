
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from parameter_loader import ParameterLoader
from LaserPulse.archive.physics_model_v5 import SpectralAmplifier

def calculate_fwhm(wavelengths, spectrum):
    """计算半高全宽 (FWHM)"""
    half_max = np.max(spectrum) / 2.0
    indices = np.where(spectrum > half_max)[0]
    if len(indices) < 2:
        return 0.0
    return (wavelengths[indices[-1]] - wavelengths[indices[0]]) * 1e9

def run_test():
    print("="*60)
    print(" 🧪 TEST 3: 增益窄化现象验证")
    print("="*60)

    # 1. 加载参数
    loader = ParameterLoader()
    cry = loader.get_crystal_params()
    seed = loader.get_seed_params()
    pump = loader.get_pump_params()
    cav = loader.get_cavity_params()

    # 2. 关键设置：关闭光谱整形 (复现黑线情况)
    cav.loss_shaping_depth = 0.0  
    print(f"⚙️  光谱整形状态: 已关闭 (Depth = {cav.loss_shaping_depth})")

    # 初始化放大器模型
    amp = SpectralAmplifier(cry, seed, pump, cav)

    # 3. 运行动力学仿真
    # 再生放大器需要一定时间建立稳态，我们模拟 500 个脉冲周期

    total_pulses = 500 
    print(f"🚀 正在运行动力学仿真 ({total_pulses} 个脉冲周期)...")
    
    result = amp.run_pulse_train_dynamics(total_pulses=total_pulses, round_trips=seed.round_trips)
    
    # 4. 结果分析
    final_pulse_energy = result['energy_trace'][-1]
    final_spectrum = result['final_spectrum']
    wavelengths = result['wavelengths']

    print(f"✅ 仿真完成。")
    print(f"📊 单脉冲输出能量: {final_pulse_energy*1e6:.2f} uJ ")
    
    # 获取种子光谱用于对比 (归一化)
    seed_spec = amp.seed_spectrum_J
    norm_seed = seed_spec / np.max(seed_spec)
    norm_out = final_spectrum / np.max(final_spectrum)

    # 计算 FWHM
    fwhm_seed = calculate_fwhm(wavelengths, seed_spec)
    fwhm_out = calculate_fwhm(wavelengths, final_spectrum)
    
    print(f"📉 种子光带宽 (FWHM):   {fwhm_seed:.2f} nm")
    print(f"📉 放大后带宽 (FWHM):   {fwhm_out:.2f} nm")
    
    if fwhm_out < fwhm_seed * 0.6:
        print("✅ 成功观测到增益窄化：带宽显著变窄。")
    else:
        print("❌ 警告：未观测到明显的增益窄化，请检查增益系数或往返次数。")

    # 5. 绘图 (模仿 Figure 2(b))
    plt.figure(figsize=(10, 6))
    
    # 绘制种子 (虚线，代表初始状态)
    plt.plot(wavelengths*1e9, norm_seed, 'k--', linewidth=1.5, alpha=0.5, label=f'Input Seed ({fwhm_seed:.1f}nm)')
    
    # 绘制放大后光谱 (实线黑线，代表无整形输出)
    plt.plot(wavelengths*1e9, norm_out, 'k-', linewidth=2.5, label=f'Amplified - No Shaping ({fwhm_out:.1f}nm)')
    
    plt.title(f'Reproduction of Fig 2(b) (Black Line): Gain Narrowing\n({seed.round_trips} Round Trips, Output: {final_pulse_energy*1e6:.1f} $\\mu$J)', fontsize=14)
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Normalized Intensity (a.u.)', fontsize=12)
    plt.xlim(1015, 1060) # 聚焦在增益峰附近
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)

    # 标注峰值
    peak_idx = np.argmax(final_spectrum)
    peak_wl = wavelengths[peak_idx] * 1e9
    plt.text(peak_wl+2, 0.9, f"Peak @ {peak_wl:.1f}nm", color='black', fontweight='bold')

    save_path = "result_step3_narrowing.png"
    plt.savefig(save_path)
    print(f"\n📈 验证曲线已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    run_test()