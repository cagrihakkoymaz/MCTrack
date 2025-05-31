import matplotlib.pyplot as plt
import pandas as pd

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
    file_labels = ["benchmark", "ego_vehicle", "gt", "experiment"]
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

# Load data
benchmark_file = "0012_benchmark.txt"
ego_vehicle_file = "0012_ego_vehicle.txt"
gt_file = "0012_gt.txt"
experiment_file = "0012.txt"

benchmark_df = read_bbox_data(benchmark_file)
ego_vehicle_df = read_bbox_data(ego_vehicle_file)
gt_df = read_bbox_data(gt_file)
experiment_df = read_bbox_data(experiment_file)

file_list = [benchmark_df, ego_vehicle_df, gt_df, experiment_df]

# Specify track ID pairs for comparison
track_ids_dict_list = [
    {"benchmark": [1], "ego_vehicle": [5], "gt": [1], "experiment": [0]},
    {"benchmark": [0], "ego_vehicle": [6], "gt": [3], "experiment": [1]}
]

plot_comparisons(track_ids_dict_list, file_list)
