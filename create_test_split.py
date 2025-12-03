"""
create test csvs by modifying train csvs to point to *001 trajectories.
this gives us a train/test split from our downloaded data.
"""

import os
import shutil

# mapping from train trajectories to test trajectories
# train csvs use: 6000, 5001, 4001, 3000, 2001, 1000, 0001
# available for test: 6001, 5000, 4000, 3001, 2000, 1001
trajectory_mapping = {
    "trajectory_6000": "trajectory_6001",  # PLE winter
    "trajectory_5001": "trajectory_5000",  # PLE spring
    "trajectory_4001": "trajectory_4000",  # PLE fall
    "trajectory_3000": "trajectory_3001",  # Kite cloudy
    "trajectory_2001": "trajectory_2000",  # Kite foggy
    "trajectory_1000": "trajectory_1001",  # Kite sunset
}

train_dir = "data/midair/train_data"
test_dir = "data/midair/test_data_custom"

# create test directory structure
os.makedirs(test_dir, exist_ok=True)

# process each train csv
for root, dirs, files in os.walk(train_dir):
    for file in files:
        if file.endswith(".csv"):
            train_csv_path = os.path.join(root, file)

            # create corresponding test directory
            rel_path = os.path.relpath(root, train_dir)
            test_subdir = os.path.join(test_dir, rel_path)
            os.makedirs(test_subdir, exist_ok=True)

            # read train csv and modify paths
            with open(train_csv_path, 'r') as f:
                content = f.read()

            # replace trajectory numbers
            modified = content
            for train_traj, test_traj in trajectory_mapping.items():
                modified = modified.replace(train_traj, test_traj)

            # check if any replacement was made
            if modified != content:
                test_csv_path = os.path.join(test_subdir, file)
                with open(test_csv_path, 'w') as f:
                    f.write(modified)
                print(f"created: {test_csv_path}")
            else:
                print(f"skipped (no matching trajectories): {train_csv_path}")

print("\ndone! test csvs created in:", test_dir)
print("\nto evaluate, run:")
print("  python main.py --mode=eval --dataset=midair --records_path=data/midair/test_data_custom --ckpt_dir=./ckpt")
