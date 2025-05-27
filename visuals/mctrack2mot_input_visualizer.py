import json
from dataclasses import dataclass, field
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from matplotlib import cm
from matplotlib.colors import Normalize
from collections import Counter

@dataclass
class SimpleStructObj:
    id: int
    x: float
    y: float
    z: float
    yaw: float
    length: float
    width: float
    height: float
    score: float
    category: str

@dataclass
class SimpleFrameMsg:
    frame_id: int
    objects: List[SimpleStructObj] = field(default_factory=list)

def plot_global_xy(scene_data, scene_id=0):
    x_vals = []
    y_vals = []

    car_x, car_y, car_yaw = [], [], []
    ped_x, ped_y, ped_yaw = [], [], []

    car_x, car_y, car_yaw = [], [], []
    ped_x, ped_y, ped_yaw = [], [], []

    for frame in scene_data:
        for obj in frame.objects:
            if obj.category == "car":
                car_x.append(obj.x)
                car_y.append(obj.y)
                car_yaw.append(obj.yaw)
            elif obj.category == "pedestrian":
                ped_x.append(obj.x)
                ped_y.append(obj.y)
                ped_yaw.append(obj.yaw)



    plt.figure(figsize=(8, 6))
    plt.scatter(car_x, car_y, c='blue', s=10, label="Car")
    plt.scatter(ped_x, ped_y, c='red', s=10, label="Pedestrian")

    # Draw arrows (heading)
    arrow_scale = 0.5  # you can tune this
    plt.quiver(car_x, car_y,
            np.cos(car_yaw), np.sin(car_yaw),
            angles='xy', scale_units='xy', scale=1/arrow_scale, color='blue', width=0.003)

    plt.quiver(ped_x, ped_y,
            np.cos(ped_yaw), np.sin(ped_yaw),
            angles='xy', scale_units='xy', scale=1/arrow_scale, color='red', width=0.003)
    plt.title(f"Global X-Y Positions (Scene ID: {scene_id})")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()


def load_prediction_txt(txt_path):
    """
    Parses output KITTI-style .txt result file for a scene into frame-wise structure.
    """
    scene = dict()
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            frame_id = int(parts[0].split('=')[1])
            track_id = int(parts[1].split('=')[1])
            category = parts[2].split('=')[1].lower()
            x = float(parts[3].split('=')[1])
            y = float(parts[4].split('=')[1])
            z = float(parts[5].split('=')[1])
            yaw = float(parts[6].split('=')[1])
            score = float(parts[7].split('=')[1])
            length = float(parts[8].split('=')[1])
            width = float(parts[9].split('=')[1])
            height = float(parts[10].split('=')[1])


            obj = SimpleStructObj(
                id=track_id,
                x=x, y=y, z=z, yaw=yaw,
                length=length, width=width, height=height,
                score=score,
                category=category
            )

            if frame_id not in scene:
                scene[frame_id] = SimpleFrameMsg(frame_id=frame_id)
            scene[frame_id].objects.append(obj)
    return [scene[i] for i in sorted(scene.keys())]

def draw_bbox(ax, x, y, yaw, l, w, color):
    trans = Affine2D().rotate_around(x, y, yaw) + ax.transData
    rect = Rectangle((x - l / 2, y - w / 2), l, w,
                     linewidth=1, edgecolor=color, facecolor='none', transform=trans)
    ax.add_patch(rect)

# Plot both
def plot_compare_input_output(input_scene, output_scene, scene_id):
    fig, ax = plt.subplots(figsize=(10, 8))

    def draw_bbox(ax, x, y, yaw, l, w, color):
        trans = Affine2D().rotate_around(x, y, yaw) + ax.transData
        rect = Rectangle((x - l / 2, y - w / 2), l, w,
                         linewidth=1, edgecolor=color, facecolor='none', transform=trans)
        ax.add_patch(rect)
    # INPUT
    for frame in input_scene:
        for obj in frame.objects:
            color = 'blue' if obj.category == "car" else 'yellow'
            ax.scatter(obj.x, obj.y, c=color, s=20, label=f"Input {obj.category.capitalize()}", marker='o')
            draw_bbox(ax, obj.x, obj.y, obj.yaw, obj.length, obj.width, color)


    # OUTPUT
    # OUTPUT - Color by ID
    # Count how often each ID appears
    id_counts = Counter(obj.id for frame in output_scene for obj in frame.objects)
    valid_ids = [obj_id for obj_id, count in id_counts.items() if count >= 30]

    if valid_ids:
        norm = Normalize(vmin=min(valid_ids), vmax=max(valid_ids))
        cmap = cm.get_cmap("nipy_spectral")  # Wide, vivid color range

    for frame in output_scene:
        for obj in frame.objects:
            if obj.id in valid_ids:
                color = cmap(norm(obj.id))
            else:
                color = "lightgray"  # default color for rare IDs
            ax.scatter(obj.x, obj.y, c=[color], s=10, marker='x', label=f"Output ID={obj.id}")
            draw_bbox(ax, obj.x, obj.y, obj.yaw, obj.length, obj.width, color)





    ax.set_title(f"Scene {scene_id}: Input vs Output")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis("equal")
    ax.grid(True)

    # De-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.show()



def load_mctrack_data(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def load_inputs(mctrack_data):
    scenes = [[] for _ in range(21)]
    idobj = 0
    for scene_id, scene in mctrack_data.items():
        scenes[int(scene_id)] = []
        frame_id = 0
        for frame in scene:
            frame_msg = SimpleFrameMsg(frame_id=frame_id)
            for bbox in frame.get("bboxes", []):
                obj = SimpleStructObj(
                    id=idobj,
                    x=bbox.get("global_xyz", [0.0, 0.0, 0.0])[0],
                    y=bbox.get("global_xyz", [0.0, 0.0, 0.0])[1],
                    z=bbox.get("global_xyz", [0.0, 0.0, 0.0])[2],
                    yaw=bbox.get("global_yaw", 0.0),
                    length=bbox.get("lwh", [0.0, 0.0, 0.0])[0],
                    width=bbox.get("lwh", [0.0, 0.0, 0.0])[1],
                    height=bbox.get("lwh", [0.0, 0.0, 0.0])[2],
                    score=bbox.get("detection_score", 0.0),
                    category=bbox.get("category", "unknown")
                )
                idobj += 1
                frame_msg.objects.append(obj)

            scenes[int(scene_id)].append(frame_msg)
            frame_id += 1

    return scenes




if __name__ == "__main__":
    

    json_path = "/home/chakkoym/kitti_ws/MCTrack/data/base_version/kitti/virconv/val.json"
    mctrack_data = load_mctrack_data(json_path)
    inputs = load_inputs(mctrack_data)
    visualize_all_scene=False
    run_time="20250415_090721"
    if(visualize_all_scene):
        for scene_id in range(len(inputs)):
            #plot_global_xy(inputs[scene_id],scene_id)
            input_scene = inputs[scene_id]

        

                # Load OUTPUT
            txt_path = f"/home/chakkoym/kitti_ws/MCTrack/results/kitti/{run_time}/virconv/val/data/{scene_id:04d}_compare.txt"
            output_scene = load_prediction_txt(txt_path)

            plot_compare_input_output(input_scene, output_scene, scene_id)
    else:    
        scene_id=15  
        input_scene = inputs[scene_id]



            # Load OUTPUT
        txt_path = f"/home/chakkoym/kitti_ws/MCTrack/results/kitti/{run_time}/virconv/val/data/{scene_id:04d}_compare.txt"
        output_scene = load_prediction_txt(txt_path)

        plot_compare_input_output(input_scene, output_scene, scene_id)