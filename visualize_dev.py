from collections import defaultdict
from typing import Dict, List, Optional
import os
import torch
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interp

from argoverse.map_representation.map_api import ArgoverseMap

_ZORDER = {"AGENT": 15, "AV": 10, "FEATS": 25, "PREDICTION": 20}


def interpolate_polyline(polyline: np.ndarray, num_points: int) -> np.ndarray:
    duplicates = []
    for i in range(1, len(polyline)):
        if np.allclose(polyline[i], polyline[i - 1]):
            duplicates.append(i)
    if polyline.shape[0] - len(duplicates) < 4:
        return polyline
    if duplicates:
        polyline = np.delete(polyline, duplicates, axis=0)
    tck, u = interp.splprep(polyline.T, s=0)
    u = np.linspace(0.0, 1.0, num_points)
    return np.column_stack(interp.splev(u, tck))


def viz_sequence(
    df: pd.DataFrame,
    lane_centerlines: Optional[List[np.ndarray]] = None,
    show: bool = True,
    smoothen: bool = True,
) -> None:

    # Seq data
    city_name = df["CITY_NAME"].values[0]

    if lane_centerlines is None:
        # Get API for Argo Dataset map
        avm = ArgoverseMap()

    plt.figure(0, figsize=(16, 14), dpi=300)
    plt.axis('off')
    plt.axis('equal')
    x_min = min(df["X"])-12
    x_max = max(df["X"])+12
    y_min = min(df["Y"])-12
    y_max = max(df["Y"])+12

    if lane_centerlines is None:

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

    frames = df.groupby("TRACK_ID")
    plt.xlabel("Map X")
    plt.ylabel("Map Y")

    color_dict = {"AGENT": "#d33e4c", "FEATS": "#d33e4c", "AV": "grey", "PREDICTION": "#0ABDFF"}
    object_type_tracker: Dict[int, int] = defaultdict(int)

    # Plot trajectories
    for group_name, group_data in frames:
        object_type = group_data["OBJECT_TYPE"].values[0]
        cor_x = group_data["X"].values
        cor_y = group_data["Y"].values

        # Check if we need to smooth the trajectories
        if smoothen:
            polyline = np.column_stack((cor_x, cor_y))
            num_points = cor_x.shape[0] * 2
            smooth_polyline = interpolate_polyline(polyline, num_points)
            cor_x = smooth_polyline[:, 0]
            cor_y = smooth_polyline[:, 1]
        if object_type == "PREDICTION":
            # Plot the predicted trajectory in cyan
            plt.plot(
                cor_x, 
                cor_y, 
                "--",
                color=color_dict[object_type],
                label=object_type if not object_type_tracker[object_type] else "",
                alpha=1,
                linewidth=2,
                zorder=_ZORDER[object_type],
                )
            plt.arrow(
                cor_x[-2], cor_y[-2], 
                cor_x[-1] - cor_x[-2], 
                cor_y[-1] - cor_y[-2], 
                color=color_dict[object_type], 
                head_width=1, 
                head_length=1.5, 
                zorder=_ZORDER[object_type]
                )
            
        elif object_type == "AGENT":
            final_x = cor_x[0]
            final_y = cor_y[0]
            # Plot the ground truth trajectory in light blue
            plt.plot(
                cor_x, 
                cor_y, 
                "--",
                color=color_dict[object_type],
                label=object_type if not object_type_tracker[object_type] else "",
                alpha=1,
                linewidth=2,
                zorder=_ZORDER[object_type],
                )
            plt.arrow(
                cor_x[-2], cor_y[-2], 
                cor_x[-1] - cor_x[-2], 
                cor_y[-1] - cor_y[-2], 
                color=color_dict[object_type], 
                head_width=1, 
                head_length=1.5, 
                zorder=_ZORDER[object_type]
                )

        elif object_type == "FEATS":
            final_x = cor_x[0]
            final_y = cor_y[0]
            # Plot the ground truth trajectory in light blue
            plt.plot(
                cor_x, 
                cor_y, 
                "--",
                color=color_dict[object_type],
                label=object_type if not object_type_tracker[object_type] else "",
                alpha=1,
                linewidth=2,
                zorder=_ZORDER[object_type],
                )
            
        elif object_type == "AV":
            final_x = cor_x[0]
            final_y = cor_y[0]
            # Plot the ground truth trajectory in light blue
            plt.plot(
                cor_x, 
                cor_y, 
                "--",
                color=color_dict[object_type],
                label=object_type if not object_type_tracker[object_type] else "",
                alpha=0.5,
                linewidth=1,
                zorder=_ZORDER[object_type],
                )
            plt.arrow(
                cor_x[-2], cor_y[-2], 
                cor_x[-1] - cor_x[-2], 
                cor_y[-1] - cor_y[-2], 
                color=color_dict[object_type], 
                head_width=0.4, 
                head_length=0.6, 
                zorder=_ZORDER[object_type]
                )


        if object_type == "AGENT":
            marker_type = "*"
            marker_size = 12
        elif object_type == "OTHERS":
            marker_type = "o"
            marker_size = 10
        elif object_type == "AV":
            marker_type = "o"
            marker_size = 8
        elif object_type == "FEATS":
            marker_type = "o"
            marker_size = 10


        plt.plot(
            final_x,
            final_y,
            marker_type,
            color=color_dict[object_type],
            label=object_type if not object_type_tracker[object_type] else "",
            alpha=0.4,
            markersize=marker_size,
            zorder=_ZORDER[object_type],
        )

        object_type_tracker[object_type] += 1
        
    if show:
        # plt.title(f'Trajectory Visualization for ID: {example_id} in {city_name}')
        base_path = '/root/lanegcn/results'
        model_path = os.path.join(base_path, model_vi)
        file_name = '{}_{}.jpeg'.format(model, example_id)
        file_path = os.path.join(model_path, file_name)
        # 确保目录存在
        os.makedirs(model_path, exist_ok=True)
        # 保存图像
        plt.savefig(file_path, format='jpeg')
        # plt.savefig(f'/root/lanegcn/results/{}/{model}_{example_id}.jpeg'.format(model), format='jpeg')  # Save the figure as JPEG
        plt.show()
    plt.close()

# 加载 .pkl 文件
model_vi = "inD"
model = "inD3_5"
file_path = '/root/lanegcn/results/{}/results.pkl'.format(model)  # 替换为你的文件路径
data = torch.load(file_path)

# 从 .pkl 文件中提取数据
preds = data['preds']
gts = data['gts']
gts2 = data['gts2']
feats = data["feats"]
cities = data['cities']

# 选择一个轨迹进行可视化
example_id = 600
pred_trajs = preds[example_id]
gt_traj = gts[example_id]
gt_traj2 = gts2[example_id]
feats = feats[example_id]
city_name = cities[example_id]

# 创建一个包含所有轨迹的 DataFrame
df_list = []
pred_traj = pred_trajs[1]
df_pred = pd.DataFrame(pred_traj, columns=['X', 'Y'])
df_pred['TRACK_ID'] = 'PRED_1'
df_pred['OBJECT_TYPE'] = 'PREDICTION'
df_pred['CITY_NAME'] = city_name
df_list.append(df_pred)

if gt_traj is not None:
    df_gt = pd.DataFrame(gt_traj, columns=['X', 'Y'])
    df_gt['TRACK_ID'] = 'GT'
    df_gt['OBJECT_TYPE'] = 'AGENT'
    df_gt['CITY_NAME'] = city_name
    df_list.append(df_gt)


for i, gt in enumerate(gt_traj2):
    gt = pd.DataFrame(gt, columns=['X', 'Y'])
    gt['TRACK_ID'] = f'gt_{i+1}'
    gt['OBJECT_TYPE'] = 'AV'
    gt['CITY_NAME'] = city_name
    df_list.append(gt)

if feats is not None:
    df_ft = pd.DataFrame(feats, columns=['X', 'Y'])
    df_ft['TRACK_ID'] = 'FEATS'
    df_ft['OBJECT_TYPE'] = 'FEATS'
    df_ft['CITY_NAME'] = city_name
    df_list.append(df_ft)

df = pd.concat(df_list)

# 可视化并保存图像
viz_sequence(df, show=True, smoothen=True)