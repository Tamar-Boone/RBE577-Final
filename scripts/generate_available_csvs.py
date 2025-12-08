'''
modified csv generator for available complete data only.
generates train/test csvs for kite_training/cloudy which has complete color + depth data.
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

if __name__ == "__main__":
    os.makedirs(a.output_dir, exist_ok=True)

    # only process environments/climates with complete data
    # format: (environment, [list of climates with complete data])
    complete_data = [
        ("Kite_training", ["cloudy"]),
        # add more as data becomes available:
        # ("Kite_training", ["sunny"]),  # when sunny stereo_disparity is complete
        # ("PLE_training", ["fall"]),     # when ple fall is complete
    ]

    sensors = [["color_left", ".JPEG"], ["stereo_disparity", ".PNG"]]

    for env, climates in complete_data:
        for climate in climates:
            print(f"Processing {env}/{climate}")

            color_path = os.path.join(a.db_path, env, climate, sensors[0][0])
            if not os.path.exists(color_path):
                print(f"  Skipping - path not found: {color_path}")
                continue

            trajectories = sorted(os.listdir(color_path))
            h5_path = os.path.join(a.db_path, env, climate, "sensor_records.hdf5")

            if not os.path.exists(h5_path):
                print(f"  Skipping - sensor records not found: {h5_path}")
                continue

            h5_db = h5py.File(h5_path, 'r')

            for traj_idx, traj in enumerate(trajectories):
                # 2:1 train/test split by trajectory index
                if traj_idx % 3 != 0:
                    out_dir = os.path.join(a.output_dir, "train_data", env, climate)
                else:
                    out_dir = os.path.join(a.output_dir, "test_data", env, climate)

                os.makedirs(out_dir, exist_ok=True)
                file_name = os.path.join(out_dir, f"traj_{str(traj_idx).zfill(4)}.csv")

                with open(file_name, 'w') as file:
                    # write csv header
                    file.write("id\tcamera_l\tdisp\tqw\tqx\tqy\tqz\ttx\tty\ttz\n")

                    def get_path(sensor, index, ext):
                        im_name = str(index).zfill(6) + "." + ext
                        return os.path.join(env, climate, sensor, traj, im_name)

                    # read groundtruth poses from hdf5
                    # 4 imu measurements between camera frames, skip by FRAME_SKIP
                    r_a = np.array(h5_db[traj]["groundtruth"]["attitude"][:-(4*FRAME_SKIP), :])
                    r_b = np.array(h5_db[traj]["groundtruth"]["attitude"][(4*FRAME_SKIP):, :])
                    p_a = np.array(h5_db[traj]["groundtruth"]["position"][:-(4*FRAME_SKIP), :])
                    p_b = np.array(h5_db[traj]["groundtruth"]["position"][(4*FRAME_SKIP):, :])

                    traj_len = r_a.shape[0] // (FRAME_SKIP * 4)

                    for index in range(traj_len):
                        # compute frame-to-frame camera motion
                        i = index * FRAME_SKIP
                        q_r_a = Quaternion(r_a[i*4, :])
                        q_r_b = Quaternion(r_b[i*4, :])
                        trans = q_r_a.conjugate.rotate(p_b[i*4, :] - p_a[i*4, :])
                        rot = (q_r_a.conjugate * q_r_b).elements

                        camera_l = get_path("color_left", i + FRAME_SKIP, "JPEG")
                        stereo_disp = get_path("stereo_disparity", i + FRAME_SKIP, "PNG")

                        # convert from body frame to camera frame
                        rot = rot.tolist()
                        rot = [rot[0], rot[2], rot[3], rot[1]]
                        trans = [trans.tolist()[1], trans.tolist()[2], trans.tolist()[0]]

                        file.write(f"{index}\t{camera_l}\t{stereo_disp}\t{rot[0]:.6f}\t{rot[1]:.6f}\t{rot[2]:.6f}\t{rot[3]:.6f}\t{trans[0]:.6f}\t{trans[1]:.6f}\t{trans[2]:.6f}\n")

                print(f"  Generated: {file_name} ({traj_len} frames)")

            h5_db.close()

    print("\nCSV generation complete!")
    print(f"Output directory: {a.output_dir}")
