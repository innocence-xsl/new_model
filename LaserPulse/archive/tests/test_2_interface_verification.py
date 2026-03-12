"""
TEST 1: 验证物理模型接口
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from parameter_loader import ParameterLoader
from LaserPulse.archive.physics_model_v5 import SpectralAmplifier

def run_test_2_interface_verification():
    print("\n" + "="*60)
    print("🚀 Test 2: 种子光预整形接口验证 (Interface Check)")
    print("="*60)   

    # 1. 初始化环境
    # 使用 align_to_peak=True 确保我们挖坑挖在正中心
    loader = ParameterLoader()
    seed_params = loader.get_seed_params(align_to_peak=True, target_center_nm=1040.0)
    
    # 获取默认参数
    amp = SpectralAmplifier(loader.get_crystal_params(), 
                            loader.get_seed_params(), 
                            loader.get_pump_params(), 
                            loader.get_cavity_params())

    # 2. 设定整形参数 (这是我们要测试的变量)
    SHAPE_DEPTH_PER_PASS = seed_params.shaping_depth        # 深度 20 %
    SHAPE_WIDTH = seed_params.shaping_width                 # 宽度 30 nm    
    SHAPE_CENTER = seed_params.shaping_center               # 中心 1040 nm
    PASSES = 7              # 衰减次数

    print(f"🔧 设定整形参数: Depth={SHAPE_DEPTH_PER_PASS:.1%}, Width={SHAPE_WIDTH}nm, Center={SHAPE_CENTER*1e9:.1f}nm, Bounce={PASSES}")
    
    # 3. 执行整形
    # 执行循环衰减
    spectrum_orig = amp.seed_spectrum_J.copy()
    spectrum_current = spectrum_orig.copy()
    
    for i in range(PASSES):
        spectrum_current = amp._apply_math_filter(
            spectrum=spectrum_current,      # 输入上一次的结果
            center=SHAPE_CENTER,            # 中心波长
            width=SHAPE_WIDTH,              # 陷波宽度
            depth=SHAPE_DEPTH_PER_PASS      # 单次深度 (20%)
        )
        
    # 更新最终结果
    spectrum_shaped = spectrum_current
    SHAPE_DEPTH = 1.0 - (1.0 - SHAPE_DEPTH_PER_PASS)**PASSES
    
    # 4. 能量分析 (Energy Analysis)
    E_orig = np.sum(amp.seed_spectrum_J) * amp.seed.seed_area
    E_shaped = np.sum(spectrum_shaped) * amp.seed.seed_area
    
    # 换算为 nJ, pJ 方便阅读
    E_orig_pJ = E_orig * 1e12
    E_shaped_pJ = E_shaped * 1e12
    loss_ratio = (E_orig - E_shaped) / E_orig

    print("-" * 40)
    print(f"📊 能量监测报告:")
    print(f"   - 原始种子能量: {E_orig_pJ:.2f} pJ")
    print(f"   - 整形后能量:   {E_shaped_pJ:.2f} pJ  <--- 关键指标")
    print(f"   - 插入损耗:     {loss_ratio:.2%} (约 -{10*np.log10(1-loss_ratio+1e-10):.1f} dB)")

    # 5. 安全阀检查 (Safety Valve Check)
    # 经验值：再生放大器注入能量建议 > 50 pJ 以压制 ASE 噪声
    SAFE_THRESHOLD_PJ = 50.0 
    
    if E_shaped_pJ < SAFE_THRESHOLD_PJ:
        print(f"\n❌ [DANGER] 警告！整形后能量低于安全阈值 ({SAFE_THRESHOLD_PJ} pJ)！")
        print("   -> 风险：种子光可能被 ASE 噪声淹没，导致输出脉冲信噪比极差。")
        print("   -> 建议：1. 减小滤波深度; 2. 增加原始种子源能量。")
    else:
        print(f"\n✅ [PASS] 安全检查通过。能量充足，可进行后续放大。")

    # 6. 可视化验证
    plt.figure(figsize=(10, 6), dpi=120)
    wl_nm = amp.wavelengths * 1e9
    
    # 为了看清形状，使用归一化坐标，但要在图例中标注真实能量
    plt.plot(wl_nm, amp.seed_spectrum_J / np.max(amp.seed_spectrum_J), 
             'k--', alpha=0.5, label='Original Seed')
    
    # 绘制整形后的谱（归一化到自身的峰值，以便观察形状）
    # 注意：如果能量太小，除法可能会报警
    if np.max(spectrum_shaped) > 0:
        plt.plot(wl_nm, spectrum_shaped / np.max(spectrum_shaped), 
                 'r-', linewidth=2, label=f'Shaped Seed\nDepth={SHAPE_DEPTH:.0%} | Loss~{loss_ratio*100:.0f}% ')
         
        # 绘制“坑”的形状 (滤波器透射曲线)
        # 我们可以反推滤波器曲线用于展示
        filter_curve = spectrum_shaped / (amp.seed_spectrum_J + 1e-20)
        # 限制显示范围防止边缘噪点
        mask = amp.seed_spectrum_J > (np.max(amp.seed_spectrum_J) * 0.01)
        plt.plot(wl_nm[mask], filter_curve[mask], 'g:', linewidth=1.5, label='Filter Transmission T($\\lambda$)')

    plt.title(f"Cavity Spectral Shaping Verification\nEnergy: {E_orig_pJ:.1f} pJ -> {E_shaped_pJ:.1f} pJ", fontsize=12)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Intensity / Transmission")
    plt.xlim(990, 1080)
    plt.ylim(0, 1.0)
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    save_path = "result_step2_interface.png"
    plt.savefig(save_path)
    print(f"\n📈 验证曲线已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_test_2_interface_verification()
