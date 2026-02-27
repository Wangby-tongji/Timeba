import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 数据
data = {
    'Feature Combination': [
        'X, Y', 'X, Y, Vel.', 'X, Y, Vel., Acc.', 'X, Y, Vel., Acc., Size', 'X, Y, Vel., Acc., Size, P.V./heading',
        'X, Y', 'X, Y, Vel.', 'X, Y, Vel., Acc.', 'X, Y, Vel., Acc., Size', 'X, Y, Vel., Acc., Size, P.V./heading',
        'X, Y', 'X, Y, Vel.', 'X, Y, Vel., Acc.', 'X, Y, Vel., Acc., Size'
    ],
    'Datasets': [
        'highD', 'highD', 'highD', 'highD', 'highD', 
        'exiD', 'exiD', 'exiD', 'exiD', 'exiD',
        'NGSIM', 'NGSIM', 'NGSIM', 'NGSIM'
    ],
    'RMSE (k=6)': [0.302, 0.292, 0.224, 0.157, 0.180, 
                   0.546, 0.448, 0.185, 0.255, 0.446,
                   1.129, 0.957, 0.956, 1.075],
    'RMSE (k=1)': [1.080, 0.524, 0.496, 0.456, 0.608, 
                   1.426, 0.813, 0.346, 0.350, 0.973,
                   2.763, 2.557, 2.452, 2.725]
}

df = pd.DataFrame(data)

# 设置Seaborn样式
sns.set(style="whitegrid")

# 分别绘制k=6和k=1的折线图
fig, ax = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

# 绘制k=6的折线图
sns.lineplot(data=df, x='Feature Combination', y='RMSE (k=6)', hue='Datasets', marker='o', ax=ax[0])
ax[0].set_title('RMSE (k=6) by Feature Combination and Dataset')
ax[0].set_ylabel('RMSE (k=6)')
ax[0].legend(title='Datasets')
ax[0].grid(True)

# 绘制k=1的折线图
sns.lineplot(data=df, x='Feature Combination', y='RMSE (k=1)', hue='Datasets', marker='o', ax=ax[1])
ax[1].set_title('RMSE (k=1) by Feature Combination and Dataset')
ax[1].set_ylabel('RMSE (k=1)')
ax[1].set_xlabel('Feature Combination')
ax[1].legend(title='Datasets')
ax[1].grid(True)

# 旋转x轴标签
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()