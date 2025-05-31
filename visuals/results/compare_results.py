import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_bbox_data(file_path):
    """Reads bounding box data from the provided text file."""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            if len(values) < 10:
                continue  # Ensure the line has enough values
            frame = int(values[0])
            track_id = int(values[1])
            bbox = list(map(float, values[6:10]))  # Extract bbox coordinates
            data.append([frame, track_id] + bbox)
    
    df = pd.DataFrame(data, columns=["Frame", "Track_ID", "Xmin", "Ymin", "Xmax", "Ymax"])
    return df

def plot_bbox_comparison(track_ids_dict_list, file_list):
    """Plots bounding box values for multiple sets of track IDs across different sources in separate windows."""
    file_labels = ["benchmark", "ego_vehicle", "gt", "experiment"]  # Match dictionary keys
    file_colors = ['r', 'g', 'b', 'm']  # Added magenta for the experiment file
    
    for track_ids_dict in track_ids_dict_list:
        sources = [(file_list[i], track_ids_dict[file_labels[i]], file_labels[i].title(), file_colors[i]) for i in range(len(file_list))]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        labels = ["Xmin", "Xmax", "Ymin", "Ymax"]
        
        for i, col in enumerate(labels):
            for df, track_ids, label, color in sources:
                for track_id in track_ids:
                    track_data = df[df['Track_ID'] == track_id]
                    if not track_data.empty:
                        axes[i].plot(track_data['Frame'], track_data[col], label=f'{label} {col} (ID {track_id})', linestyle='dashed', color=color)
            axes[i].set_xlabel("Frame")
            axes[i].set_ylabel(f"Bounding Box {col}")
            axes[i].set_title(f"{col} Comparison for Specified Track IDs")
            axes[i].legend()
            axes[i].grid(True)
        
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

# Specify multiple sets of track IDs for separate windows
track_ids_dict_list = [
    {"benchmark": [1], "ego_vehicle": [5], "gt": [1], "experiment": [0]},
    {"benchmark": [0], "ego_vehicle": [6], "gt": [3], "experiment": [1]}
]

plot_bbox_comparison(track_ids_dict_list, file_list)
