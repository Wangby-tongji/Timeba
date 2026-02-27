import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 数据
data = {
    'Multivariate Feature': [
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

# 将数据转换为适合绘制热力图的格式
df_pivot_k6 = df.pivot(index="Multivariate Feature", columns="Datasets", values="RMSE (k=6)")
df_pivot_k1 = df.pivot(index="Multivariate Feature", columns="Datasets", values="RMSE (k=1)")

# 设置Seaborn样式
sns.set(style="whitegrid")

# 分别绘制k=6和k=1的热力图
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# 绘制k=6的热力图
sns.heatmap(df_pivot_k6, annot=True, fmt=".3f", cmap='Reds', ax=ax[0])
ax[0].set_title('按多变量特征和数据集绘制的RMSE(k=6)热力图', fontproperties="SimHei")
ax[0].set_xlabel('', fontproperties="SimHei")
ax[0].set_ylabel('多变量特征', fontproperties="SimHei")

# 绘制k=1的热力图
sns.heatmap(df_pivot_k1, annot=True, fmt=".3f", cmap='Reds', ax=ax[1])
ax[1].set_title('按多变量特征和数据集绘制的RMSE(k=1)热力图', fontproperties="SimHei")
ax[1].set_xlabel('数据集', fontproperties="SimHei")
ax[1].set_ylabel('多变量特征', fontproperties="SimHei")

plt.tight_layout()
plt.show()