# seed_spectrum.py
# 种子光谱参数诊断工具 - 终极版
# 包含：几何插值法、RMS统计法、高斯拟合法 + 中心波长对比

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit  # 引入拟合工具

# 导入项目模块
from archive.physics_model_v5 import SpectralAmplifier
from system.parameter_loader import ParameterLoader

# # 设置绘图风格
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
# plt.rcParams['axes.unicode_minus'] = False

# --- 算法 1: 几何插值法 (原有) ---
def calculate_fwhm_geometric(wavelengths, intensity):
    max_val = np.max(intensity)
    half_max = max_val / 2.0
    signs = np.sign(intensity - half_max)
    diff = np.diff(signs)
    crossings = np.where(diff != 0)[0]
    
    crossing_wls = []
    for idx in crossings:
        y1 = intensity[idx]
        y2 = intensity[idx+1]
        x1 = wavelengths[idx]
        x2 = wavelengths[idx+1]
        if y2 != y1:
            x_cross = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
            crossing_wls.append(x_cross)
            
    if len(crossing_wls) >= 2:
        return crossing_wls[-1] - crossing_wls[0], (crossing_wls[-1] + crossing_wls[0])/2
    return 0.0, wavelengths[np.argmax(intensity)]

# --- 算法 2: RMS 统计法 (新加入) ---
def calculate_fwhm_rms(wavelengths, intensity):
    """
    计算二阶矩（标准差），适用于不规则形状
    FWHM ≈ 2.355 * sigma
    """
    # 归一化分布函数
    prob_dist = intensity / np.sum(intensity)
    
    # 1. 计算中心波长 (一阶矩)
    mean_wl = np.sum(wavelengths * prob_dist)
    
    # 2. 计算方差 (二阶矩)
    variance = np.sum(((wavelengths - mean_wl) ** 2) * prob_dist)
    sigma = np.sqrt(variance)
    
    # 3. 转换为 FWHM
    fwhm_rms = 2.3548 * sigma
    return fwhm_rms, mean_wl

# --- 算法 3: 整体高斯拟合法 (新加入) ---
def gaussian_func(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def calculate_fwhm_fit(wavelengths, intensity):
    try:
        # 初始猜测: 峰值高度, 重心位置, 粗略宽度
        p0 = [np.max(intensity), np.mean(wavelengths), (wavelengths[-1]-wavelengths[0])/6]
        
        # 执行拟合
        popt, _ = curve_fit(gaussian_func, wavelengths, intensity, p0=p0)
        
        a_fit, x0_fit, sigma_fit = popt
        fwhm_fit = 2.3548 * abs(sigma_fit)
        
        # 生成拟合曲线用于绘图
        fit_curve = gaussian_func(wavelengths, *popt)
        return fwhm_fit, x0_fit, fit_curve
    except Exception as e:
        print(f"拟合失败: {e}")
        return 0.0, 0.0, np.zeros_like(wavelengths)

def debug_seed_loading():
    print("🔍 [诊断模式] 正在进行全方位光谱分析...\n")
    
    loader = ParameterLoader()
    seed = loader.get_seed_params()
    cry = loader.get_crystal_params()
    pump = loader.get_pump_params()
    cav = loader.get_cavity_params()
    
    amp = SpectralAmplifier(cry, seed, pump, cav)
    spec = amp.seed_spectrum_J
    wl = amp.wavelengths
    max_val = np.max(spec)
    
    # --- 执行三种计算 ---
    fwhm_geo, center_geo = calculate_fwhm_geometric(wl, spec)
    fwhm_rms, center_rms = calculate_fwhm_rms(wl, spec)
    fwhm_fit, center_fit, spec_fit = calculate_fwhm_fit(wl, spec)
    
    target_fwhm = 27.0
    target_center = 1040.0
    
    print(f"   📊 [带宽计算方法对比]")
    print(f"      1. 几何截断法 (Geometric): {fwhm_geo*1e9:.2f} nm (受毛刺影响大)")
    print(f"      2. RMS统计法 (Statistical): {fwhm_rms*1e9:.2f} nm (物理能量分布)")
    print(f"      3. 高斯拟合法 (Curve Fit): {fwhm_fit*1e9:.2f} nm (整体最佳近似)")
    
    # --- 判定逻辑 ---
    # 选择最接近 Test Report 的方法作为参考
    errors = {
        "Geo": abs(fwhm_geo*1e9 - target_fwhm),
        "RMS": abs(fwhm_rms*1e9 - target_fwhm),
        "Fit": abs(fwhm_fit*1e9 - target_fwhm)
    }
    best_method = min(errors, key=errors.get)
    print(f"\n   ✅ [推荐] 建议参考 '{best_method}' 结果，它最接近测试报告的 {target_fwhm} nm。")

    # --- 绘图 ---
    plt.figure(figsize=(12, 7))
    
    # 1. 绘制原始数据
    plt.plot(wl*1e9, spec/max_val, 'b-', linewidth=2, label='Real Data (Simulation)', alpha=0.6)
    
    # 2. 绘制拟合数据
    plt.plot(wl*1e9, spec_fit/np.max(spec_fit), 'r--', linewidth=2, label=f'Gaussian Fit (FWHM={fwhm_fit*1e9:.1f}nm)')
    
    # 3. 绘制中心波长标记
    # 实际中心 (基于拟合或RMS)
    actual_center_nm = center_fit * 1e9
    plt.axvline(x=actual_center_nm, color='red', linestyle='-.', alpha=0.8, label=f'Actual Center: {actual_center_nm:.1f} nm')
    
    # 目标中心 (1040nm)
    plt.axvline(x=target_center, color='green', linestyle='-', linewidth=2, label=f'Target Center: {target_center:.0f} nm')
    
    # 添加标注箭头
    plt.annotate('', xy=(actual_center_nm, 1.05), xytext=(target_center, 1.05),
                 arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
    plt.text((actual_center_nm + target_center)/2, 1.07, f"Δ = {abs(actual_center_nm - target_center):.1f} nm", 
             ha='center', color='purple', fontweight='bold')

    plt.title(f"Spectral Analysis: Bandwidth & Center Wavelength\nTarget: {target_center}nm / {target_fwhm}nm", fontsize=14)
    plt.xlim(990,1080)
    plt.xlabel("Wavelength (nm)", fontsize=12)
    plt.ylabel("Normalized Intensity", fontsize=12)
    plt.legend(loc='upper right', framealpha=0.9, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 调整视角
    plt.xlim(target_center - 40, target_center + 40)
    plt.ylim(0, 1.15) # 留出顶部画箭头
    
    save_name = "result_step1_seed_spectrum.png"
    plt.savefig(save_name, dpi=150)
    print(f"\n   -> 诊断图已保存至 {save_name}")

if __name__ == "__main__":
    debug_seed_loading()