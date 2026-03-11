import numpy as np

# 定义时间带宽积常数 K
K_GAUSSIAN = 0.441  # 高斯型脉冲
K_SECH2 = 0.315     # 双曲正割平方型脉冲

# 光速 (m/s)
C = 2.99792458e8

def calculate_ftl_pulse_duration():
    """计算傅里叶变换极限脉宽 (FTL)。"""

    print("--- 傅里叶变换极限脉宽计算器 (FTL) ---")
    print("本程序使用近似公式：τ_min ≈ K * (λ₀² / (c * Δλ))")
    print("请注意所有输入必须为正值。")
    print("------------------------------------------")

    # 1. 选择脉冲形状 (K 值)
    while True:
        shape_choice = input("请选择脉冲形状类型：\n1. 高斯型 (K=0.441)\n2. sech²型 (K=0.315)\n请选择 (1 或 2): ").strip()
        if shape_choice == '1':
            K = K_GAUSSIAN
            shape_name = "高斯型"
            break
        elif shape_choice == '2':
            K = K_SECH2
            shape_name = "sech²型"
            break
        else:
            print("输入无效，请重新输入 1 或 2。")

    # 2. 输入中心波长 λ₀
    while True:
        try:
            # 约定输入单位为 nm
            lambda_nm = float(input(f"请输入中心波长 λ₀ (nm, 例如 800): ").strip())
            if lambda_nm <= 0:
                raise ValueError
            # 转换为米 (m)
            lambda_m = lambda_nm * 1e-9
            break
        except ValueError:
            print("输入无效，请输入一个正数。")

    # 3. 输入光谱半高全宽 Δλ
    while True:
        try:
            # 约定输入单位为 nm
            delta_lambda_nm = float(input(f"请输入光谱半高全宽 Δλ (nm, 例如 10): ").strip())
            if delta_lambda_nm <= 0:
                raise ValueError
            # 转换为米 (m)
            delta_lambda_m = delta_lambda_nm * 1e-9
            break
        except ValueError:
            print("输入无效，请输入一个正数。")

    # 4. 计算 FTL 脉宽 τ_min
    try:
        # 公式: τ_min ≈ K * (λ₀² / (c * Δλ))
        tau_min_seconds = K * (lambda_m**2 / (C * delta_lambda_m))

        # 将结果转换为飞秒 (fs)
        tau_min_fs = tau_min_seconds * 1e15

        # 5. 输出结果
        print("\n--- 计算结果 ---")
        print(f"所选脉冲形状: {shape_name} (K = {K})")
        print(f"中心波长 λ₀: {lambda_nm:.2f} nm")
        print(f"光谱带宽 Δλ: {delta_lambda_nm:.2f} nm")
        print("----------------")
        print(f"⭐ 傅里叶变换极限脉宽 τ_min: {tau_min_fs:.2f} fs")
        print("----------------")

    except Exception as e:
        print(f"计算发生错误: {e}")

# 运行程序
if __name__ == "__main__":
    calculate_ftl_pulse_duration()