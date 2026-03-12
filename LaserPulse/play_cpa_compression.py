# play_cpa_compression.py
import matplotlib.pyplot as plt
import numpy as np
import copy

from system.parameter_loader import ParameterLoader
from components.active_components import AdvancedBulkCrystal
from components.passive_components import LossyMirror, SpectralFilter, AdvancedMaterialDispersion, TreacyGratingCompressor
from core.dataclasses_def import PhysicalConstants
from system.optical_assembly import RegenerativeAmplifierAssembly

def main():
    print("🌟 CPA 物理终极沙盒：直面超快激光之癌 🌟\n")
    
    loader = ParameterLoader()
    cry_params, pump_params, cav_params = loader.get_crystal_params(), loader.get_pump_params(), loader.get_cavity_params()
    seed_pulse, seed_params = loader.get_seed_params()

    # 1. 腔外预整形：打造宽光谱
    pre_shaper = SpectralFilter(
        name="Pre_Shaper", center_wl=1040e-9, width=15e-9, 
        depth_per_bounce=1-(1-0.8)**(1/7), num_bounces=7)
    seed_pulse = pre_shaper.propagate(seed_pulse)

    # 2. 组装真实色散腔体（包含 Beta2 和 Beta3）
    crystal = AdvancedBulkCrystal(
        name="Yb:CALGO_Crystal", crystal_params=cry_params, seed_params=seed_params,
        pump_params=pump_params, consts=PhysicalConstants(), pump_polarization='pi')
    crystal.pump_crystal(dt=10e-3)
    
    mirror = LossyMirror(name="Mirror", reflectivity=1.0 - cav_params.loss_passive)
    
    # ⚠️ 加入真实的晶体色散
    # Yb:CALGO 在 1040nm 附近典型的正色散
    cry_length = cry_params.thickness
    crystal_beta2 = 30e-27  # 30,000 fs^2/mm
    crystal_beta3 = 40e-42  # 40,000 fs^3/mm
    
    real_dispersion = AdvancedMaterialDispersion(
        name="Crystal_Dispersion", length=cry_length, 
        beta2=crystal_beta2, beta3=crystal_beta3)

    regen_amp = RegenerativeAmplifierAssembly(
        name="Real_CPA_Amp", cavity_components=[crystal, mirror, real_dispersion])
    
    # 3. 开始放大：马拉松长跑
    num_round_trips = 70
    print(f"🚀 开始真实的跑圈！总计 {num_round_trips} 圈...")
    uncompressed_pulse = regen_amp.simulate(seed_pulse=seed_pulse, num_round_trips=num_round_trips)
    
    # 计算晶体中累积的巨大总色散
    total_accumulated_beta2 = num_round_trips * cry_length * crystal_beta2
    total_accumulated_beta3 = num_round_trips * cry_length * crystal_beta3

    # 4. 进入真实光栅压缩器
    print(f"\n🗜️ 离开腔体！此时累积的晶体 Beta 3 达到了: {total_accumulated_beta3*1e42:.2f} fs^3")
    print("🗜️ 正在启动真实衍射光栅压缩器...")
    
    # 准备一对典型的 1000 lines/mm，45度入射角的光栅
    compressor = TreacyGratingCompressor(
        name="Treacy_Compressor", groove_density_mm=1000, input_angle_deg=45)
    
    # 使用厂长特权，让光栅自动寻找抵消晶体 Beta 2 的距离！
    compressor.auto_tune_separation(-total_accumulated_beta2, uncompressed_pulse.grid.omega_c)
    
    compressed_pulse = compressor.propagate(copy.deepcopy(uncompressed_pulse))

    # 5. 计算极限脉宽 (FTL) 以作对比
    ftl_pulse = copy.deepcopy(compressed_pulse)
    ftl_pulse.A_f = np.abs(ftl_pulse.A_f) + 0j
    ftl_pulse.to_time_domain()

    # ==========================================
    # 画图：见证丑陋但真实的“底座”
    # ==========================================
    print("\n📊 正在生成物理真相图表...")
    fig, ax = plt.subplots(figsize=(10, 6))

    time_fs = np.linspace(-ftl_pulse.grid.points/2, ftl_pulse.grid.points/2 - 1, ftl_pulse.grid.points) * (1.0 / (ftl_pulse.grid.points * ftl_pulse.grid.df)) * 1e15

    I_real = np.abs(compressed_pulse.A_t)**2
    I_real /= np.max(I_real)
    
    I_ftl = np.abs(ftl_pulse.A_t)**2
    I_ftl /= np.max(I_ftl)

    # 绘制理想的极限脉冲 (黑色虚线)
    ax.plot(time_fs, I_ftl, color='black', linestyle='--', linewidth=2, label='Ideal Transform Limit (0 Phase)')
    
    # 绘制被真实物理摧残后的脉冲 (红色实线，带底座)
    ax.plot(time_fs, I_real, color='red', linewidth=2.5, label='Real Grating Compression (with $\\beta_3$ mismatch)')
    ax.fill_between(time_fs, I_real, alpha=0.2, color='red')

    ax.set_title("The Reality of CPA: Uncompensated Higher-Order Dispersion", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (fs)", fontsize=12)
    ax.set_ylabel("Normalized Intensity", fontsize=12)
    ax.set_xlim(-1000, 1000) # 看清 2000 飞秒内的全部细节
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=12, loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()