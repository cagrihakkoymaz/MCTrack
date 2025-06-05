# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import json, yaml
import logging
import copy
import argparse
import os
import time
import multiprocessing
from tqdm import tqdm
from datetime import datetime
from functools import partial
from tracker.base_tracker import Base3DTracker
from dataset.baseversion_dataset import BaseVersionTrackingDataset
from evaluation.static_evaluation.kitti.evaluation_HOTA.scripts.run_kitti import (
    eval_kitti,
)
import sys

from evaluation.static_evaluation.nuscenes.eval import eval_nusc
from evaluation.static_evaluation.waymo.eval import eval_waymo
from utils.kitti_utils import save_results_kitti
from utils.nusc_utils import save_results_nuscenes, save_results_nuscenes_for_motion
from utils.waymo_utils.convert_result import save_results_waymo
from playsound import playsound


def run(scene_id, scenes_data, cfg, args, tracking_results):
    ##print("--------Running  START-------------")

    """
    Info: This function tracks objects in a given scene, processes frame data, and stores tracking results.
    Parameters:
        input:
            scene_id: ID of the scene to process.
            scenes_data: Dictionary with scene data.
            cfg: Configuration settings for tracking.
            args: Additional arguments.
            tracking_results: Dictionary to store results.
        output:
            tracking_results: Updated tracking results for the scene.
    """
    scene_data = scenes_data[scene_id]
    ##print("--------SCENE DATA  START-------------")
    ##print(len(scene_data))
    #print(type(scene_data))
    #print(type(scene_data[0]))
    #print((scene_data[0]))
    #print((scene_data[2]))

    ##print("--------SCENE DATA END-------------")

    dataset = BaseVersionTrackingDataset(scene_id, scene_data, cfg=cfg)
    tracker = Base3DTracker(cfg=cfg)
    all_trajs = {}

    for index in tqdm(range(len(dataset)), desc=f"Processing {scene_id}"):
        frame_info = dataset[index]
        frame_id = frame_info.frame_id
        print("frame_id : ",frame_id)
        cur_sample_token = frame_info.cur_sample_token
        all_traj = tracker.track_single_frame(frame_info)
        result_info = {
            "frame_id": frame_id,
            "cur_sample_token": cur_sample_token,
            "trajs": copy.deepcopy(all_traj),
            "transform_matrix": frame_info.transform_matrix,
        }
        all_trajs[frame_id] = copy.deepcopy(result_info)
    if cfg["TRACKING_MODE"] == "GLOBAL":
        trajs = tracker.post_processing()
        for index in tqdm(
            range(len(dataset)), desc=f"Trajectory Postprocessing {scene_id}"
        ):
            frame_id = dataset[index].frame_id
            for track_id in sorted(list(trajs.keys())):
                for bbox in trajs[track_id].bboxes:
                    if (
                        bbox.frame_id == frame_id
                        and bbox.is_interpolation
                        and track_id not in all_trajs[frame_id]["trajs"].keys()
                    ):
                        all_trajs[frame_id]["trajs"][track_id] = bbox

        for index in tqdm(
            range(len(dataset)), desc=f"Trajectory Postprocessing {scene_id}"
        ):
            frame_id = dataset[index].frame_id
            for track_id in sorted(list(trajs.keys())):
                det_score = 0
                for bbox in trajs[track_id].bboxes:
                    det_score = bbox.det_score
                    break
                if (
                    track_id in all_trajs[frame_id]["trajs"].keys()
                    and det_score <= cfg["THRESHOLD"]["GLOBAL_TRACK_SCORE"]
                ):
                    del all_trajs[frame_id]["trajs"][track_id]
    print("scene id",scene_id)
    #print("all_trajs",all_trajs)
    tracking_results[scene_id] = all_trajs
    #plot_imm_probabilities(tracker.all_trajs)
    plot_state_estimates(tracker.all_trajs)
    results = compute_rmse(tracker.all_trajs)
    for tid, vals in results.items():
        print(f"[Track {tid} | Frames: {vals['count']}]")
        print(f"  🔹 RMSE Pred: X={vals['rmse_pred'][0]:.3f}, Y={vals['rmse_pred'][1]:.3f}")
        print(f"  🔸 RMSE Updt: X={vals['rmse_upd'][0]:.3f}, Y={vals['rmse_upd'][1]:.3f}\n")
import matplotlib.pyplot as plt

def plot_imm_probabilities(traj_dict, title_prefix="IMM Probabilities"):
    """
    traj_dict: Dictionary of {track_id: Trajectory}
    """
    for track_id, traj in traj_dict.items():
        if not hasattr(traj, "imm_prob_history") or not traj.imm_prob_history:
            continue

        frames, cv_probs, ctra_probs = zip(*traj.imm_prob_history)

        plt.figure(figsize=(10, 4))
        plt.plot(frames, cv_probs, label='CV Probability', color='blue')
        plt.plot(frames, ctra_probs, label='CTRA Probability', color='orange')
        plt.title(f"{title_prefix} - Track ID {track_id}")
        plt.xlabel("Frame")
        plt.ylabel("Probability")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_state_estimates(traj_dict, total_frames=371, title_prefix="State Estimates"):
    """
    traj_dict: Dictionary of {track_id: Trajectory}
    Plots X and Y position estimates (measured, updated, predicted) over time and trajectory in one figure.
    """
    for track_id, traj in traj_dict.items():
        if not traj.measurements:
            continue

        full_frames = list(range(total_frames))
        meas_x = [np.nan] * total_frames
        meas_y = [np.nan] * total_frames
        upd_x = [np.nan] * total_frames
        upd_y = [np.nan] * total_frames
        pred_x = [np.nan] * total_frames
        pred_y = [np.nan] * total_frames

        used_frames = [bbox.frame_id for bbox in traj.bboxes[-len(traj.measurements):]]
        #pred_x = [p[0] for p in traj.motion_updates]
        #pred_y = [p[1] for p in traj.motion_updates]

        for idx, frame_id in enumerate(used_frames):
            if frame_id < total_frames:


                meas_x[frame_id] = traj.measurements[idx][0]
                meas_y[frame_id] = traj.measurements[idx][1]
                upd_x[frame_id] = traj.measurement_updates[idx][0]
                upd_y[frame_id] = traj.measurement_updates[idx][1]
                pred_x[frame_id] = traj.motion_updates[idx][0]
                pred_y[frame_id] = traj.motion_updates[idx][1]
                """
                print(f"\nTrack ID: {track_id}")
                print(f"  - len(measurements): {len(traj.measurements)}")
                print(f"  - measurements:")
                for i, m in enumerate(traj.measurements):
                    print(f"    [{i}] {m}")

                print(f"  - len(measurement_updates): {len(traj.measurement_updates)}")
                print(f"  - measurement_updates:")
                for i, u in enumerate(traj.measurement_updates):
                    print(f"    [{i}] {u}")

                print(f"  - len(motion_updates): {len(traj.motion_updates)}")
                print(f"  - motion_updates:")
                for i, p in enumerate(traj.motion_updates):
                    print(f"    [{i}] {p}")

                print(f"  - len(bboxes): {len(traj.bboxes)}")
                print(f"  - bboxes frame_ids:")
                for i, b in enumerate(traj.bboxes):
                    print(f"    [{i}] frame_id: {b.frame_id}")
                """

        # --- Create 3 subplots in one figure ---
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        # X vs Frame
        axs[0].plot(full_frames, meas_x, 'o-', label='Meas X', alpha=0.7)
        axs[0].plot(full_frames, upd_x, 's-', label='Update X', alpha=0.7)
        axs[0].plot(full_frames[:len(pred_x)], pred_x, 'x--', label='Predict X', alpha=0.7)
        axs[0].set_title(f"{title_prefix} (X) - Track ID {track_id}")
        axs[0].set_xlabel("Frame")
        axs[0].set_ylabel("X Position")
        axs[0].grid(True)
        axs[0].legend()

        # Y vs Frame
        axs[1].plot(full_frames, meas_y, 'o-', label='Meas Y', alpha=0.7)
        axs[1].plot(full_frames, upd_y, 's-', label='Update Y', alpha=0.7)
        axs[1].plot(full_frames[:len(pred_y)], pred_y, 'x--', label='Predict Y', alpha=0.7)
        axs[1].set_title(f"{title_prefix} (Y) - Track ID {track_id}")
        axs[1].set_xlabel("Frame")
        axs[1].set_ylabel("Y Position")
        axs[1].grid(True)
        axs[1].legend()

        # XY Trajectory
        axs[2].plot(meas_x, meas_y, 'o-', label='Measurement XY', alpha=0.6)
        axs[2].plot(upd_x, upd_y, 's-', label='Update XY', alpha=0.6)
        axs[2].plot(pred_x, pred_y, 'x--', label='Prediction XY', alpha=0.6)

        axs[2].set_title(f"2D XY Trajectory - Track ID {track_id}")
        axs[2].set_xlabel("X Position")
        axs[2].set_ylabel("Y Position")
        axs[2].grid(True)
        axs[2].axis("equal")
        axs[2].legend()

        plt.tight_layout()
        plt.show()


import numpy as np

def compute_rmse(all_trajs):
    """
    Computes RMSE for each object (track_id) between measurements and:
      - motion predictions
      - measurement updates

    Returns:
        Dictionary in the form:
        {
            track_id: {
                'rmse_pred': [rmse_x, rmse_y],
                'rmse_upd':  [rmse_x, rmse_y],
                'count': N
            },
            ...
        }
    """
    rmse_results = {}

    for track_id, traj in all_trajs.items():
        if not traj.measurements:
            continue

        # Truncate to equal length
        N = min(len(traj.measurements), len(traj.motion_updates), len(traj.measurement_updates))
        meas = np.array([m[:2] for m in traj.measurements[:N]])
        pred = np.squeeze(np.array(traj.motion_updates[:N]))
        upd  = np.squeeze(np.array(traj.measurement_updates[:N]))

        if meas.shape != pred.shape or meas.shape != upd.shape:
            print(f"⚠️ Shape mismatch for track {track_id}, skipping.")
            continue

        mse_pred = np.mean((pred - meas) ** 2, axis=0)
        mse_upd  = np.mean((upd  - meas) ** 2, axis=0)

        rmse_results[track_id] = {
            "rmse_pred": np.sqrt(mse_pred),
            "rmse_upd":  np.sqrt(mse_upd),
            "count": N
        }

    return rmse_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCTrack")
    parser.add_argument(
        "--dataset",
        type=str,
        default="kitti",
        help="Which Dataset: kitti/nuscenes/waymo",
    )
    parser.add_argument("--eval", "-e", action="store_true", help="evaluation")
    parser.add_argument("--load_image", "-lm", action="store_true", help="load_image")
    parser.add_argument("--load_point", "-lp", action="store_true", help="load_point")
    parser.add_argument("--debug", action="store_true", help="debug")
    parser.add_argument("--mode", "-m", action="store_true", help="online or offline")
    parser.add_argument("--process", "-p", type=int, default=1, help="multi-process!")
    args = parser.parse_args()

    if args.dataset == "kitti":
        cfg_path = "./config/kitti.yaml"
    elif args.dataset == "nuscenes":
        cfg_path = "./config/nuscenes.yaml"
    elif args.dataset == "waymo":
        cfg_path = "./config/waymo.yaml"
    if args.mode:
        cfg_path = cfg_path.replace(".yaml", "_offline.yaml")

    cfg = yaml.load(open(cfg_path, "r"), Loader=yaml.Loader)

    save_path = os.path.join(
        os.path.dirname(cfg["SAVE_PATH"]),
        cfg["DATASET"],
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(save_path, exist_ok=True)
    cfg["SAVE_PATH"] = save_path

    start_time = time.time()

    detections_root = os.path.join(
        cfg["DETECTIONS_ROOT"], cfg["DETECTOR"], cfg["SPLIT"] + ".json"
    )
    #print(cfg)
    with open(detections_root, "r", encoding="utf-8") as file:
        print(f"Loading data from {detections_root}...")
        data = json.load(file)
        print("Data loaded successfully.")

    if args.debug:
        if args.dataset == "kitti":
            scene_lists = [str(scene_id).zfill(4) for scene_id in cfg["TRACKING_SEQS"]]
        elif args.dataset == "nuscenes":
            scene_lists = [scene_id for scene_id in data.keys()][:2]
        else:
            scene_lists = [scene_id for scene_id in data.keys()][:2]
    else:
        scene_lists = [scene_id for scene_id in data.keys()]
        scene_lists = [str(scene_id).zfill(4) for scene_id in cfg["TRACKING_SEQS"]]
    """
    print("--------------------scene list   start-----------------------------------------------")
    print(type(scene_lists))
    print(len(scene_lists))
    print(type(scene_lists[0]))

    print(len(scene_lists[0]))
    print((scene_lists[0]))
    print((scene_lists[1]))
    print((scene_lists[2]))


    print("--------------------scene list end-----------------------------------------------")    
    """

    manager = multiprocessing.Manager()
    tracking_results = manager.dict()
    if args.process > 1:
        pool = multiprocessing.Pool(args.process)
        func = partial(
            run, scenes_data=data, cfg=cfg, args=args, tracking_results=tracking_results
        )
        pool.map(func, scene_lists)
        pool.close()
        pool.join()
    else:
        for scene_id in tqdm(scene_lists, desc="Running scenes"):
            print("************************************************************************")
            print("scene id ",scene_id)
            run(scene_id, data, cfg, args, tracking_results)

    for scene_id in tqdm(scene_lists, desc="Running scenes"):
            """
            print("--------SCENE DATA ID : -------------")
            print(scene_id)            
            #print("--------Data-------------")
            print(type(data))
            print(len(data))
            #print((data.keys()))
            #print(type(data[scene_id]))
            print(len(data[scene_id]))
            #print(type(data[scene_id][0]))
            print(len(data[scene_id][0]))
            print(tracking_results)
            """

    tracking_results = dict(tracking_results)
    #print(tracking_results,"tracking results:")
    if args.dataset == "kitti":
        save_results_kitti(tracking_results, cfg)
        if args.eval:
            eval_kitti(cfg)
    if args.dataset == "nuscenes":
        save_results_nuscenes(tracking_results, save_path)
        save_results_nuscenes_for_motion(tracking_results, save_path)
        if args.eval:
            eval_nusc(cfg)
    elif args.dataset == "waymo":
        save_results_waymo(tracking_results, save_path)
        if args.eval:
            eval_waymo(cfg, save_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
