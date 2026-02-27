import matplotlib.pyplot as plt
import numpy as np

# 数据
datasets = ['NGSIM-4-512', 'NGSIM-4-256', 'NGSIM-4-128']
metrics = ['minADE', 'minFDE', 'MR', 'RMSE']
K_values = [6, 1]

data = {
    'NGSIM-4-512': {
        6: {'minADE': 0.8496885680894835, 'minFDE': 1.489518310098678, 'MR': 0.19346256176358798, 'RMSE': 0.955508444712094},
        1: {'minADE': 1.9089271069592166, 'minFDE': 4.79247982134562, 'MR': 0.7111364500190042, 'RMSE': 2.451562110571446}
    },
    'NGSIM-4-256': {
        6: {'minADE': 0.9659919581674418, 'minFDE': 1.8652322470641618, 'MR': 0.35499809958190803, 'RMSE': 1.1113431735670418},
        1: {'minADE': 2.059161698366298, 'minFDE': 5.202819518543922, 'MR': 0.7635879893576587, 'RMSE': 2.641002264230486}
    },
    'NGSIM-4-128': {
        6: {'minADE': 0.9682787899648309, 'minFDE': 1.8703728168322307, 'MR': 0.33447358418852147, 'RMSE': 1.1151888106598318},
        1: {'minADE': 2.0526358575713997, 'minFDE': 5.140065004814009, 'MR': 0.7620676548840745, 'RMSE': 2.6239625524113044}
    }
}

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]
    for dataset in datasets:
        values_6 = data[dataset][6][metric]
        values_1 = data[dataset][1][metric]
        ax.plot(K_values, [values_6, values_1], marker='o', label=dataset)
    
    ax.set_title(metric)
    ax.set_xlabel('Max #guesses (K)')
    ax.set_ylabel(metric)
    ax.set_xticks(K_values)
    ax.legend()

plt.tight_layout()
plt.savefig('metrics2.png')
plt.show()