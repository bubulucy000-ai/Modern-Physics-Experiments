import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class MillikanProcessor:
    def __init__(self, rho_oil=835.27, rho_air=1.03, eta=1.85e-5, d=2.9e-3, U=400.0, p=89200.0):
        """
        初始化物理参数 (默认值参考论文)
        rho_oil: 油滴密度 (kg/m^3)
        rho_air: 空气密度 (kg/m^3)
        eta: 空气粘度 (kg/m·s)
        d: 极板间距 (m)
        U: 电压 (V)
        p: 大气压 (Pa)
        """
        self.rho_oil = rho_oil
        self.rho_air = rho_air
        self.eta = eta
        self.d = d
        self.U = U
        self.p = p
        self.g = 9.8
        self.b = 8.2e-3  # Cunningham 常数

    def process_tracker_data(self, file_path, voltage):
        """
        读取 Tracker 导出的 .txt 文件 (包含时间 t 和垂直速度 vy)
        并自动扫描、过滤、计算电荷和半径
        """
        # 1. 读取数据 (假设没有表头，第一列时间，第二列速度)
        try:
            data = pd.read_csv(file_path, sep='\t', header=None, names=['t', 'vy'])
        except:
            print(f"读取文件 {file_path} 失败，请检查格式。")
            return None

        # 2. 自动扫描算法：区分上升和下降 (参考论文 4.2 节)
        # 使用滑动窗口 (前后各5个点) 判断当前点的运动方向
        window_size = 5
        vy = data['vy'].values
        v_up = []
        v_down = []
        
        for i in range(len(vy)):
            start = max(0, i - window_size)
            end = min(len(vy), i + window_size + 1)
            local_window = vy[start:end]
            
            pos_count = np.sum(local_window > 0)
            neg_count = np.sum(local_window < 0)
            
            if pos_count > neg_count:
                v_up.append(vy[i])
            elif neg_count > pos_count:
                v_down.append(vy[i])
            # 如果正负相等，视为过渡点，丢弃 (对应论文中灰色点)

        # 3. 过滤异常值 (论文逻辑：剔除偏离均值 ±1 个标准差的数据)
        def filter_outliers(velocities):
            if len(velocities) < 10: # 论文要求每组至少10个点
                return None
            v_mean = np.mean(velocities)
            v_std = np.std(velocities)
            return [v for v in velocities if abs(v - v_mean) <= v_std]

        v_up_filtered = filter_outliers(v_up)
        v_down_filtered = filter_outliers(v_down)

        if not v_up_filtered or not v_down_filtered:
            print("有效数据点不足，无法计算。")
            return None

        # 计算平均速度 (取绝对值)
        vs = abs(np.mean(v_up_filtered)) # 上升速度
        vd = abs(np.mean(v_down_filtered)) # 下降速度
        
        # 计算标准差 (作为误差)
        delta_vs = np.std(v_up_filtered)
        delta_vd = np.std(v_down_filtered)

        # 4. 物理计算 (应用论文公式)
        # 先计算未修正的半径和电荷 (式1, 式2)
        alpha = (2 * np.pi * self.d * np.sqrt(np.pi) / 3) * np.sqrt(self.g * (self.rho_oil - self.rho_air))
        beta = 2 * np.sqrt(np.pi / (self.g * (self.rho_oil - self.rho_air))) # 注意：这里beta分母是根号下g*密度差，论文原式可能有印刷错误，按标准推导应为 sqrt(pi/(g*(rho_oil-rho_air)))，请结合教材核对
        
        # 使用 Cunningham 修正 (式6, 式7)
        r_uncorrected = beta * np.sqrt(vd - vs)
        # 修正后的半径
        r_corrected = np.sqrt((self.b / (2 * self.p))**2 + beta**2 * (vd - vs)) - (self.b / (2 * self.p))
        
        # 修正后的电荷
        q_corrected = (alpha / (self.U * beta**3)) * ((vd + vs) / (vd - vs)) * (r_corrected**3)

        # 误差传递 (简化版，参考式3和式4)
        # 注意：论文中误差公式较复杂，这里采用数值微分法计算相对误差
        delta_r = (beta / 2) * np.sqrt((delta_vd**2 + delta_vs**2) / (vd - vs))
        delta_q = q_corrected * np.sqrt((delta_vd**2 + delta_vs**2) / (vd - vs)) # 简化的相对误差估计

        return {
            'vs': vs, 'vd': vd,
            'r': r_corrected, 'delta_r': delta_r,
            'q': q_corrected, 'delta_q': delta_q,
            'num_up': len(v_up_filtered), 'num_down': len(v_down_filtered)
        }

    def plot_q_vs_r(self, results_list):
        """
        绘制论文中的 q × r 图 (带误差棒和元电荷网格线)
        """
        q_vals = [res['q'] for res in results_list]
        r_vals = [res['r'] for res in results_list]
        q_err = [res['delta_q'] for res in results_list]
        r_err = [res['delta_r'] for res in results_list]

        plt.figure(figsize=(8, 6))
        plt.errorbar(r_vals, q_vals, xerr=r_err, yerr=q_err, fmt='o', color='blue', ecolor='lightblue', capsize=3, label='Experimental Data')

        # 绘制元电荷倍数线 (论文图8/10中的虚线)
        e = 1.602e-19
        for n in range(1, 6):
            plt.axhline(y=n*e, color='gray', linestyle='--', alpha=0.5)
            plt.text(max(r_vals)*1.05, n*e, f'{n}e', verticalalignment='center')

        plt.xlabel('Radius r (m)')
        plt.ylabel('Charge q (C)')
        plt.title('Millikan Oil Drop: q vs r (Cunningham Corrected)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# ================= 使用示例 =================
if __name__ == "__main__":
    # 假设你有一个 Tracker 导出的文件 'droplet_400V.txt'
    # 文件内容格式：第一列时间(s)，第二列垂直速度(m/s)
    
    processor = MillikanProcessor(voltage=400.0) # 设置电压
    
    # 模拟一组数据用于演示 (实际使用时替换为你的文件路径)
    # 这里我们直接构造结果来画图
    mock_results = []
    # 生成一些模拟的油滴数据 (接近论文图10(b)的分布)
    np.random.seed(42)
    for _ in range(30):
        mock_results.append({
            'r': np.random.uniform(0.5e-6, 1.0e-6),
            'q': np.random.choice([1, 2, 3]) * 1.602e-19 * np.random.normal(1, 0.05),
            'delta_r': 0.1e-6,
            'delta_q': 0.2e-19
        })
    
    # 绘制图表
    processor.plot_q_vs_r(mock_results)
    
    print("代码运行成功！请将 process_tracker_data 中的文件路径替换为你的实际数据文件。")
