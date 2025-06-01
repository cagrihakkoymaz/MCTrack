import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def read_bbox_data(file_path):
    """Reads bounding box, location, rotation, and dimension data from the file."""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            if len(values) < 17:
                continue  # Ensure the line has enough values
            frame = int(values[0])
            track_id = int(values[1])
            bbox = list(map(float, values[6:10]))          # [Xmin, Ymin, Xmax, Ymax]
            dims = list(map(float, values[10:13]))          # [height, width, length]
            loc = list(map(float, values[13:16]))           # [loc_x, loc_y, loc_z]
            rot = float(values[16])                         # rotation_y
            data.append([frame, track_id] + bbox + dims + loc + [rot])
    
    df = pd.DataFrame(data, columns=[
        "Frame", "Track_ID",
        "Xmin", "Ymin", "Xmax", "Ymax",
        "Dim_Height", "Dim_Width", "Dim_Length",
        "Loc_X", "Loc_Y", "Loc_Z",
        "Rotation_Y"
    ])
    return df

def plot_comparisons(track_ids_dict_list, file_list):
    """Plots BBox, Location/Rotation, and Dimensions for specified track IDs."""
    file_labels = ["cv", "ctra", "gt", "imm"]
    file_colors = ['r', 'g', 'b', 'm']
    
    for track_ids_dict in track_ids_dict_list:
        sources = [
            (file_list[i], track_ids_dict[file_labels[i]], file_labels[i].title(), file_colors[i])
            for i in range(len(file_list))
        ]

        # --- 2x2 BBox Plot ---
        fig_bbox, axes_bbox = plt.subplots(2, 2, figsize=(12, 8))
        axes_bbox = axes_bbox.flatten()
        bbox_labels = ["Xmin", "Xmax", "Ymin", "Ymax"]
        
        for i, col in enumerate(bbox_labels):
            for df, track_ids, label, color in sources:
                for track_id in track_ids:
                    track_data = df[df['Track_ID'] == track_id]
                    if not track_data.empty:
                        axes_bbox[i].plot(track_data['Frame'], track_data[col], label=f'{label} {col} (ID {track_id})', linestyle='dashed', color=color)
            axes_bbox[i].set_xlabel("Frame")
            axes_bbox[i].set_ylabel(col)
            axes_bbox[i].set_title(f"BBox {col} Comparison")
            axes_bbox[i].legend()
            axes_bbox[i].grid(True)

        plt.tight_layout()
        plt.show()

        # --- 2x2 Location & Rotation Plot ---
        fig_loc, axes_loc = plt.subplots(2, 2, figsize=(12, 8))
        axes_loc = axes_loc.flatten()
        loc_labels = ["Loc_X", "Loc_Y", "Loc_Z", "Rotation_Y"]
        
        for i, col in enumerate(loc_labels):
            for df, track_ids, label, color in sources:
                for track_id in track_ids:
                    track_data = df[df['Track_ID'] == track_id]
                    if not track_data.empty:
                        axes_loc[i].plot(track_data['Frame'], track_data[col], label=f'{label} {col} (ID {track_id})', linestyle='dashed', color=color)
            axes_loc[i].set_xlabel("Frame")
            axes_loc[i].set_ylabel(col)
            axes_loc[i].set_title(f"{col} Comparison")
            axes_loc[i].legend()
            axes_loc[i].grid(True)

        plt.tight_layout()
        plt.show()

        # --- 1x3 Dimension Plot ---
        fig_dim, axes_dim = plt.subplots(1, 3, figsize=(14, 4))
        dim_labels = ["Dim_Height", "Dim_Width", "Dim_Length"]
        
        for i, col in enumerate(dim_labels):
            for df, track_ids, label, color in sources:
                for track_id in track_ids:
                    track_data = df[df['Track_ID'] == track_id]
                    if not track_data.empty:
                        axes_dim[i].plot(track_data['Frame'], track_data[col], label=f'{label} {col} (ID {track_id})', linestyle='dashed', color=color)
            axes_dim[i].set_xlabel("Frame")
            axes_dim[i].set_ylabel(col)
            axes_dim[i].set_title(f"{col} Comparison")
            axes_dim[i].legend()
            axes_dim[i].grid(True)

        plt.tight_layout()
        plt.show()

from sklearn.metrics import mean_squared_error
from math import sqrt


def compute_rmse_by_track_id(pred_df, gt_df, label):
    """
    Computes RMSE between prediction and GT by matching both Track_ID and Frame.
    """
    rmse_results = {}
    columns_to_compare = [
        "Loc_X", "Loc_Y", "Loc_Z",
        "Rotation_Y",
        "Xmin", "Ymin", "Xmax", "Ymax",
        "Dim_Height", "Dim_Width", "Dim_Length"
    ]

    pred_id = pred_df['Track_ID'].iloc[0]
    gt_id = gt_df['Track_ID'].iloc[0]


    # Frame-wise merge on Frame and Track_ID
    merged = pd.merge(pred_df, gt_df, on=["Frame"], suffixes=('', '_gt'))
    print(merged.head(1).to_string(index=False))
    if merged.empty:
        print(f"\n⚠️ No matching frames for Track_ID {pred_id} vs GT {gt_id}")
        return {}

    for col in columns_to_compare:
        pred_values = merged[col]
        gt_values = merged[f"{col}_gt"]
        rmse = np.sqrt(((gt_values - pred_values) ** 2).mean())
        rmse_results[col] = rmse

    print(f"\n📈 RMSE Results for {label} (Frame-aligned, Track_ID={pred_id}):")
    for key, val in rmse_results.items():
        print(f"  {key:12s}: {val:.4f}")

    return rmse_results

def compute_model_rmse_table_per_object(track_ids_dict_list, cv_df, ctra_df, imm_df, gt_df):
    models = {
        "IMM": imm_df,
        "CV": cv_df,
        "CTRA": ctra_df
    }

    metrics = [
        "Loc_X", "Loc_Y", "Loc_Z", "Rotation_Y",
        "Xmin", "Ymin", "Xmax", "Ymax",
        "Dim_Height", "Dim_Width", "Dim_Length"
    ]

    all_tables = []

    for obj_idx, track_ids in enumerate(track_ids_dict_list):

        gt_id = track_ids["gt"][0]
        gt_track = gt_df[gt_df["Track_ID"] == gt_id]
        model_results = []
        for model_name, model_df in models.items():
            model_id = track_ids[model_name.lower()][0]
            pred_track = model_df[model_df["Track_ID"] == model_id]

            # Frame-wise merge on Frame and Track_ID
            merged = pd.merge(pred_track, gt_track, on=["Frame"], suffixes=('', '_gt'))

            row = {"Model": model_name}
            if merged.empty:
                print(f"⚠️ No matching frames for {model_name} Track_ID {model_id} vs GT Track_ID {gt_id}")
                for metric in metrics:
                    row[metric] = None
            else:
                for metric in metrics:
                    rmse = np.sqrt(((merged[metric] - merged[f"{metric}_gt"]) ** 2).mean())
                    row[metric] = rmse

            model_results.append(row)

        result_df = pd.DataFrame(model_results)
        all_tables.append((f"Object {model_id} (GT_ID={gt_id})", result_df))

    return all_tables


# Load data
cv_file = "0012_cv.txt"
ctra_file = "0012_ctra.txt"
gt_file = "0012_gt.txt"
imm_file = "0012_imm.txt"

cv_df = read_bbox_data(cv_file)
ctra_df = read_bbox_data(ctra_file)
gt_df = read_bbox_data(gt_file)
imm_df = read_bbox_data(imm_file)

file_list = [cv_df, ctra_df, gt_df, imm_df]

# Specify track ID pairs for comparison
track_ids_dict_list = [
    {"cv": [1], "ctra": [1], "gt": [3], "imm": [1]},
    {"cv": [0], "ctra": [0], "gt": [1], "imm": [0]}
]

plot_comparisons(track_ids_dict_list, file_list)

gt_df_single = gt_df[gt_df["Track_ID"] == 1]
exp_df_single = imm_df[imm_df["Track_ID"] == 0]

#compute_rmse_by_track_id(exp_df_single, gt_df_single, label="Experiment vs GT")

rmse_tables = compute_model_rmse_table_per_object(track_ids_dict_list, cv_df, ctra_df, imm_df, gt_df)

# Print each table
for obj_name, df in rmse_tables:
    print(f"\n📊 RMSE Table for {obj_name}:")
    print(df.round(4).to_string(index=False))
