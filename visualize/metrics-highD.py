import matplotlib.pyplot as plt
import numpy as np

# 数据
methods = ["CV", "S-LSTM [46]", "CS-LSTM [47]", "NLS-LSTM [49]", "S-GAN [48]", "DLM [50]", "STDAN [31]", "MMnTP [51]", "Timeba"]
time_steps = ["1s", "2s", "3s", "4s", "5s"]
rmse_values = [
    [0.09, 0.32, 0.67, 1.14, 1.73],
    [0.22, 0.62, 1.27, 2.15, 3.41],
    [0.22, 0.61, 1.24, 2.10, 3.27],
    [0.20, 0.57, 1.14, 1.90, 2.91],
    [0.30, 0.78, 1.46, 2.34, 3.41],
    [0.22, 0.61, 1.16, 1.80, 2.80],
    [0.19, 0.27, 0.48, 0.91, 1.66],
    [0.19, 0.38, 0.62, 0.95, 1.39],
    [0.05, 0.07, 0.12, 0.26, 0.47]  # Timeba的数据
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
# plt.title('highD数据集上的RMSE评价结果', fontproperties="SimHei", fontsize=16)
# plt.xlabel('时间步', fontproperties="SimHei", fontsize=14)
plt.title('Comparison of RMSE of Different Methods on highD', fontsize=16)
plt.xlabel('Time Steps', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.ylim(0, 4)  # 调整纵轴范围，更高的比例
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 保存图像
plt.savefig('highD.png', dpi=300)

# 显示图形
plt.show()
