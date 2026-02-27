import matplotlib.pyplot as plt
import numpy as np

# 数据
methods = ["CV", "MMnTP [51]", "Timeba"]
time_steps = ["1s", "2s", "3s", "4s", "5s"]
rmse_values = [
    [0.25, 0.63, 1.19, 1.92, 2.82],
    [0.26, 0.57, 0.98, 1.50, 2.11],
    [0.04, 0.10, 0.13, 0.17, 0.35],
]

# 定义颜色，从红色到蓝色，增加饱和度差异
base_colors = [
    '#d62728',  # 红色
    # '#2ca02c',  # 绿色
    # '#ff7f0e',  # 橙色
    # '#d4b000',  # 黄色
    # '#9467bd',  # 紫色
    '#8c564b',  # 棕色
    # '#e377c2',  # 粉色
    # '#7f7f7f',  # 灰色
    '#1f77b4'   # 蓝色
]

# 创建图形
plt.figure(figsize=(10, 8))

# 绘制每种方法的曲线
for i, (method, rmse) in enumerate(zip(methods, rmse_values)):
    alpha = 1 - 0.02 * i  # 每个方法的饱和度相差5%
    plt.plot(time_steps, rmse, marker='o', label=method, color=base_colors[i], linewidth=2, markersize=8, alpha=alpha)

# 添加标题和标签
plt.title('Comparison of RMSE of Different Methods on exiD', fontsize=16)
plt.xlabel('Time Steps', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.ylim(0, 3)  # 调整纵轴范围，更高的比例
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 保存图像
plt.savefig('exiD.png', dpi=300)

# 显示图形
plt.show()
