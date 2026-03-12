


import matplotlib.pyplot as plt
import numpy as np

# 导入我们刚刚整理好的各个部门的“得力干将”
from system.parameter_loader import ParameterLoader
from components.active_components import AdvancedBulkCrystal
from components.passive_components import LossyMirror, SpectralFilter
from system.optical_assembly import RegenerativeAmplifierAssembly
from core.dataclasses_def import PhysicalConstants

def main():

    # ==========================================
    # 第一步：进货（读取所有的物理参数和初始种子光）
    # ==========================================
    print("📦 读取参数...")
    loader = ParameterLoader()
    
    # 依次拿到晶体、泵浦、腔和种子光的参数图纸
    cry_params = loader.get_crystal_params()
    pump_params = loader.get_pump_params()
    cav_params = loader.get_cavity_params()
    
    # 特别注意：获取种子光时，进货专员不仅给了参数，还直接把捏好的“初始脉冲跑车(Pulse)”交给了我们！
    seed_pulse, seed_params = loader.get_seed_params()
    
    print(f"✅ 初始种子光能量: {seed_pulse.get_energy():.2e} J")

# 🌟🌟🌟 完美复现 Wang et al. 2021 的腔外预整形 🌟🌟🌟
    if seed_params.shaping_depth > 0:
        print(f"\n🔪 正在启动腔外种子光预整形 (Multi-bounce Pre-shaping)...")
        
        # --- 物理参数反推 ---
        total_depth = seed_params.shaping_depth  # 比如文献中的 0.8 (80%)
        num_bounces = 7                          # 文献中的 7 次反射
        
        # 如果中心被挖掉了 80%，说明只剩下 20% (0.2) 透过去了
        target_center_transmission = 1.0 - total_depth
        
        # 那么单次反射透过去的比例就是 0.2 的 7 次方根
        single_bounce_transmission = target_center_transmission ** (1.0 / num_bounces)
        
        # 反推单次挖坑的深度
        depth_per_bounce = 1.0 - single_bounce_transmission
        
        print(f"   -> 目标总挖坑深度: {total_depth*100:.1f}%, 反射次数: {num_bounces} 次")
        print(f"   -> 自动推算单次反射深度: {depth_per_bounce*100:.1f}%")
        
        # 召唤升级版的整形机
        pre_shaper = SpectralFilter(
            name="Extra_Cavity_Pre_Shaper",
            center_wl=seed_params.shaping_center,
            width=seed_params.shaping_width,
            depth_per_bounce=depth_per_bounce,
            num_bounces=num_bounces
        )
        
        # 让种子光穿过这台机器 (机器内部会自动计算 7 次方的累加效应)
        seed_pulse = pre_shaper.propagate(seed_pulse)
        print(f"✅ 预整形完成！整形后注入能量: {seed_pulse.get_energy():.2e} J")
    # 🌟🌟🌟 新增结束 🌟🌟🌟
    
    # ==========================================
    # 第二步：采购和安装机器
    # ==========================================
    
    # 1. 安装核心发动机（你亲手移植的V8引擎！）
    consts = PhysicalConstants()
    crystal = AdvancedBulkCrystal(
        name="Yb:CALGO_Crystal", 
        crystal_params=cry_params, 
        seed_params=seed_params, 
        pump_params=pump_params, 
        consts=consts,
        pump_polarization='pi'
    )
    
    # 2. 安装一个损耗镜（模拟腔内的固有损耗）
    # 如果腔的被动损耗是 2% (0.02)，那这面镜子的反射率就是 0.98
    mirror = LossyMirror(
        name="Cavity_Loss_Mirror", 
        reflectivity=1.0 - cav_params.loss_passive
    )

    # ==========================================
    # 第三步：给晶体充能（踩一脚油门！）
    # ==========================================
    print("\n⚡ 正在开启泵浦源，给晶体充能...")
    # 在种子光到来之前，我们先开着泵浦光照射一段时间（比如 1 毫秒 = 1e-3 秒）
    # 这会让晶体积攒大量的上能级粒子数，准备好被提取
    crystal.pump_crystal(dt=1e-3)
    print(f"✅ 充能完毕！当前晶体上能级粒子数密度: {crystal.N_upper:.2e} m^-3")

    # ==========================================
    # 第四步：组装流水线，开始跑圈！
    # ==========================================
    # 现在脉冲每跑一圈，都会经历：晶体放大 -> 镜面损耗 -> 滤波器挖坑
    cavity_components = [crystal, mirror]
    
    # 成立总控台
    regen_amp = RegenerativeAmplifierAssembly(
        name="My_First_Regen_Amp", 
        cavity_components=cavity_components
    )
    
    # 厂长下达命令：让这个初始脉冲，在腔里跑 50 圈！
    final_pulse = regen_amp.simulate(seed_pulse=seed_pulse, num_round_trips=100)

    # ==========================================
    # 第五步：验收成果
    # ==========================================
    print("\n🎉 试车圆满结束！")
    print(f"🏆 最终输出脉冲能量: {final_pulse.get_energy():.2e} J")

# ==========================================
    # 第六步：大屏幕数据可视化 (三联屏终极版)
    # ==========================================
    print("\n📊 正在生成总控室大屏幕汇报图表...")

    # 升级：创建一块宽16、高5的巨大画板，并劈成 3 块屏幕 (1行3列)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # ------------------------------------------
    # 📺 屏幕 1：能量暴涨曲线 (Energy vs. Round Trips)
    # ------------------------------------------
    round_trips = regen_amp.history['round_trip']
    energies = regen_amp.history['pulse_energy']

    ax1.plot(round_trips, energies, marker='o', color='red', linestyle='-', markersize=4)
    ax1.set_title("Pulse Energy Growth", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Round Trip Number", fontsize=12)
    ax1.set_ylabel("Energy (Joules)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # ------------------------------------------
    # 📺 屏幕 2：最终输出光谱 (Final Spectrum)
    # ------------------------------------------
    wavelengths_nm = final_pulse.grid.lambda_window * 1e9
    final_spectrum = np.abs(final_pulse.A_f)**2

    ax2.plot(wavelengths_nm, final_spectrum, color='blue', linewidth=2)
    ax2.set_title("Final Output Spectrum", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Wavelength (nm)", fontsize=12)
    ax2.set_ylabel("Intensity (a.u.)", fontsize=12)
    ax2.set_xlim(1010, 1060)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # ------------------------------------------
    # 📺 屏幕 3：极限压缩后的时域脉冲 (FTL Time Domain)
    # ------------------------------------------
    # 见证奇迹的核心操作：我们假设腔外光栅对完美补偿了所有的相位积累！
    # 直接把复振幅的相位强行清零 (纯实数)，这就是物理上的 FTL 状态
    final_pulse.A_f = np.abs(final_pulse.A_f) + 0j 
    
    # 时空穿梭：命令跑车从频域返回时域！
    final_pulse.to_time_domain()
    
    # 自动计算极其微小的时间标尺 (单位：飞秒 fs)
    points = final_pulse.grid.points
    df = final_pulse.grid.df
    dt = 1.0 / (points * df)
    time_fs = np.linspace(-points/2, points/2 - 1, points) * dt * 1e15
    
    # 拿到纯粹的时域强度
    intensity_t = np.abs(final_pulse.A_t)**2
    intensity_t_norm = intensity_t / np.max(intensity_t) # 归一化方便看图

    # 自动算出脉冲的半高全宽 (FWHM)
    half_max = 0.5
    indices = np.where(intensity_t_norm >= half_max)[0]
    fwhm_fs = time_fs[indices[-1]] - time_fs[indices[0]]
    
    # 把激动的战果打印在终端上！
    print(f"✨ 极限脉冲宽度 (FTL FWHM): {fwhm_fs:.1f} fs ✨")

    ax3.plot(time_fs, intensity_t_norm, color='purple', linewidth=2)
    
    # 在图里把飞秒数字标出来
    ax3.fill_between(time_fs, intensity_t_norm, alpha=0.3, color='purple')
    ax3.axvspan(time_fs[indices[0]], time_fs[indices[-1]], color='yellow', alpha=0.3, label=f'FWHM: {fwhm_fs:.1f} fs')
    
    ax3.set_title("Transform-Limited Pulse", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Time (fs)", fontsize=12)
    ax3.set_ylabel("Normalized Intensity", fontsize=12)
    ax3.set_xlim(-500, 500) # 只看最核心的中心 1000 飞秒
    ax3.legend(loc="upper right", fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)

    # 自动调整排版，展示三联屏！
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()