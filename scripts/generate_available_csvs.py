'''
modified csv generator that verifies files exist before adding to csv.
only generates entries for frames where both color and disparity files exist.
based on original midair-split-generator.py from m4depth repo.
'''

import os
import argparse
import h5py
from pyquaternion import Quaternion
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--db_path", default=os.path.join(dir_path, "..", "..", "MidAir"), help="path to midair data")
parser.add_argument("--output_dir", default=os.path.join(dir_path, "..", "data", "midair"), help="output directory for csv files")
a = parser.parse_args()

FRAME_SKIP = 4  # downsample framerate (4 imu measurements between camera frames)
MIN_FRAMES = 100  # minimum frames required for a trajectory to be included


def count_files(directory, extension):
    """count files with given extension in directory"""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.endswith(extension)])


def get_max_frame_index(directory, extension):
    """get the highest frame index available in directory"""
    if not os.path.exists(directory):
        return -1
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    if not files:
        return -1
    # extract frame numbers from filenames like "000123.JPEG"
    indices = []
    for f in files:
        try:
            idx = int(f.split('.')[0])
            indices.append(idx)
        except ValueError:
            continue
    return max(indices) if indices else -1


if __name__ == "__main__":
    os.makedirs(a.output_dir, exist_ok=True)

    # process all environments and climates that exist
    # format: (environment, [list of climates])
    complete_data = [
        ("Kite_training", ["cloudy", "sunny", "foggy", "sunset"]),
        ("PLE_training", ["fall", "spring", "winter"]),
    ]

    total_train = 0
    total_test = 0

    for env, climates in complete_data:
        for climate in climates:
            print(f"Processing {env}/{climate}")

            color_path = os.path.join(a.db_path, env, climate, "color_left")
            disp_path = os.path.join(a.db_path, env, climate, "stereo_disparity")

            if not os.path.exists(color_path):
                print(f"  Skipping - color_left not found")
                continue

            if not os.path.exists(disp_path):
                print(f"  Skipping - stereo_disparity not found")
                continue

            h5_path = os.path.join(a.db_path, env, climate, "sensor_records.hdf5")
            if not os.path.exists(h5_path):
                print(f"  Skipping - sensor_records.hdf5 not found")
                continue

            # filter out .DS_Store and non-trajectory directories
            color_trajs = set([t for t in os.listdir(color_path)
                               if t.startswith("trajectory_") and not t.startswith(".")])
            disp_trajs = set([t for t in os.listdir(disp_path)
                              if t.startswith("trajectory_") and not t.startswith(".")])

            # only process trajectories that exist in both color and disparity
            common_trajs = sorted(color_trajs & disp_trajs)
            if not common_trajs:
                print(f"  Skipping - no common trajectories between color and disparity")
                continue

            print(f"  Found {len(common_trajs)} trajectories with both color and disparity")

            h5_db = h5py.File(h5_path, 'r')

            for traj_idx, traj in enumerate(common_trajs):
                # check if trajectory exists in hdf5
                if traj not in h5_db:
                    print(f"    Skipping {traj} - not in sensor_records.hdf5")
                    continue

                # get actual file counts for this trajectory
                traj_color_path = os.path.join(color_path, traj)
                traj_disp_path = os.path.join(disp_path, traj)

                max_color_idx = get_max_frame_index(traj_color_path, ".JPEG")
                max_disp_idx = get_max_frame_index(traj_disp_path, ".PNG")

                # use the minimum of both to ensure all frames exist
                max_safe_idx = min(max_color_idx, max_disp_idx)

                if max_safe_idx < FRAME_SKIP * 2:
                    print(f"    Skipping {traj} - too few frames ({max_safe_idx})")
                    continue

                # calculate safe trajectory length based on actual files
                # we need frame index (i + FRAME_SKIP) to exist, so max index is max_safe_idx - FRAME_SKIP
                safe_traj_len = (max_safe_idx - FRAME_SKIP) // FRAME_SKIP

                if safe_traj_len < MIN_FRAMES:
                    print(f"    Skipping {traj} - only {safe_traj_len} usable frames (need {MIN_FRAMES})")
                    continue

                # 2:1 train/test split by trajectory index
                if traj_idx % 3 != 0:
                    out_dir = os.path.join(a.output_dir, "train_data", env, climate)
                    total_train += safe_traj_len
                else:
                    out_dir = os.path.join(a.output_dir, "test_data", env, climate)
                    total_test += safe_traj_len

                os.makedirs(out_dir, exist_ok=True)
                file_name = os.path.join(out_dir, f"traj_{str(traj_idx).zfill(4)}.csv")

                # read groundtruth poses from hdf5
                r_a = np.array(h5_db[traj]["groundtruth"]["attitude"][:-(4*FRAME_SKIP), :])
                r_b = np.array(h5_db[traj]["groundtruth"]["attitude"][(4*FRAME_SKIP):, :])
                p_a = np.array(h5_db[traj]["groundtruth"]["position"][:-(4*FRAME_SKIP), :])
                p_b = np.array(h5_db[traj]["groundtruth"]["position"][(4*FRAME_SKIP):, :])

                written_frames = 0
                with open(file_name, 'w') as file:
                    file.write("id\tcamera_l\tdisp\tqw\tqx\tqy\tqz\ttx\tty\ttz\n")

                    for index in range(safe_traj_len):
                        i = index * FRAME_SKIP
                        frame_idx = i + FRAME_SKIP

                        # double-check files exist (belt and suspenders)
                        color_file = os.path.join(traj_color_path, f"{frame_idx:06d}.JPEG")
                        disp_file = os.path.join(traj_disp_path, f"{frame_idx:06d}.PNG")

                        if not os.path.exists(color_file) or not os.path.exists(disp_file):
                            continue

                        # compute frame-to-frame camera motion
                        if i*4 >= r_a.shape[0]:
                            break

                        q_r_a = Quaternion(r_a[i*4, :])
                        q_r_b = Quaternion(r_b[i*4, :])
                        trans = q_r_a.conjugate.rotate(p_b[i*4, :] - p_a[i*4, :])
                        rot = (q_r_a.conjugate * q_r_b).elements

                        camera_l = os.path.join(env, climate, "color_left", traj, f"{frame_idx:06d}.JPEG")
                        stereo_disp = os.path.join(env, climate, "stereo_disparity", traj, f"{frame_idx:06d}.PNG")

                        # convert from body frame to camera frame
                        rot = rot.tolist()
                        rot = [rot[0], rot[2], rot[3], rot[1]]
                        trans = [trans.tolist()[1], trans.tolist()[2], trans.tolist()[0]]

                        file.write(f"{written_frames}\t{camera_l}\t{stereo_disp}\t{rot[0]:.6f}\t{rot[1]:.6f}\t{rot[2]:.6f}\t{rot[3]:.6f}\t{trans[0]:.6f}\t{trans[1]:.6f}\t{trans[2]:.6f}\n")
                        written_frames += 1

                print(f"    Generated: {file_name} ({written_frames} frames)")

            h5_db.close()

    print(f"\nCSV generation complete!")
    print(f"Output directory: {a.output_dir}")
    print(f"Total training frames: {total_train}")
    print(f"Total test frames: {total_test}")
