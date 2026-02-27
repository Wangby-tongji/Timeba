import matplotlib.pyplot as plt
import numpy as np

# 数据
methods = ["CV", "S-LSTM [46]", "CS-LSTM [47]", "NLS-LSTM [49]", "S-GAN [48]", "DLM [50]", "STDAN [31]", "MMnTP [51]", "Timeba"]
time_steps = ["1s", "2s", "3s", "4s", "5s"]
rmse_values = [
    [0.73, 1.78, 3.13, 4.78, 6.68],
    [0.65, 1.31, 2.16, 3.25, 4.55],
    [0.58, 1.26, 2.07, 3.09, 3.98],
    [0.56, 1.22, 2.02, 3.03, 4.30],
    [0.57, 1.32, 2.22, 3.26, 4.40],
    [0.41, 0.95, 1.72, 2.64, 3.87],
    [0.42, 1.01, 1.69, 2.56, 3.67],
    [0.36, 0.96, 1.69, 2.56, 3.55],
    [0.23, 0.62, 1.12, 1.74, 2.45]  # Timeba的数据
]

# 定义颜色，从红色到蓝色，增加饱和度差异
base_colors = [
    '#d62728',  # 红色
    '#2ca02c',  # 绿色
    '#ff7f0e',  # 橙色
    '#d4b000',  # 黄色
    '#9467bd',  # 紫色
    '#8c564b',  # 棕色
    '#e377c2',  # 粉色
    '#7f7f7f',  # 灰色
    '#1f77b4'   # 蓝色
]

# 创建图形
plt.figure(figsize=(10, 8))

# 绘制每种方法的曲线
for i, (method, rmse) in enumerate(zip(methods, rmse_values)):
    alpha = 1 - 0.02 * i  # 每个方法的饱和度相差5%
    plt.plot(time_steps, rmse, marker='o', label=method, color=base_colors[i], linewidth=2, markersize=8, alpha=alpha)

# 添加标题和标签
plt.title('Comparison of RMSE of Different Methods on NGSIM', fontsize=16)
plt.xlabel('Time Steps', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.ylim(0, 8)  # 调整纵轴范围，更高的比例
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 保存图像
plt.savefig('NGSIM.png', dpi=300)

# 显示图形
plt.show()
