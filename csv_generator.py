import os
import pandas as pd
from scipy.spatial.transform import Rotation as R

# ================= CONFIGURATION =================
# Path to the folder containing the color images (camera_l)
IMAGE_DIR1 = '/home/tamar/RBE577_ws/Final/Group_Project/datasets/UseGeo/Dataset-1/undistorted_images'
IMAGE_DIR2 = '/home/tamar/RBE577_ws/Final/Group_Project/datasets/UseGeo/Dataset-2/undistorted_images'
IMAGE_DIR3 = '/home/tamar/RBE577_ws/Final/Group_Project/datasets/UseGeo/Dataset-3/undistorted_images'

# Path to the folder containing the disparity/depth maps (disp)
DISP_DIR1 = '/home/tamar/RBE577_ws/Final/Group_Project/datasets/UseGeo/Dataset-1/depth_maps'
DISP_DIR2 = '/home/tamar/RBE577_ws/Final/Group_Project/datasets/UseGeo/Dataset-2/depth_maps'
DISP_DIR3 = '/home/tamar/RBE577_ws/Final/Group_Project/datasets/UseGeo/Dataset-3/depth_maps'

# Path to your .xyz file
XYZ_FILE_PATH1 = '/home/tamar/RBE577_ws/Final/Group_Project/data/use_geo/dataset-1/Image_orientations_dataset1.xyz'
XYZ_FILE_PATH2 = '/home/tamar/RBE577_ws/Final/Group_Project/data/use_geo/dataset-2/Image_orientations_dataset2.xyz'
XYZ_FILE_PATH3 = '/home/tamar/RBE577_ws/Final/Group_Project/data/use_geo/dataset-3/Image_orientations_dataset3.xyz'

# Output filename
OUTPUT_CSV1 = 'dataset1_trajectory.csv'
OUTPUT_CSV2 = 'dataset2_trajectory.csv'
OUTPUT_CSV3 = 'dataset3_trajectory.csv'

# Extension suffix for the depth map files
DEPTH_SUFFIX = '_depth_res.tiff'

# String to remove from the image filename to match the .xyz label (which is '_res')
FILENAME_SUFFIX_TO_REMOVE = '_res'
# =================================================

def euler_to_quaternion(omega, phi, kappa):
    """
    Converts Euler angles (degrees) to Quaternion (qw, qx, qy, qz).
    """
    rot = R.from_euler('xyz', [omega, phi, kappa], degrees=True)
    x, y, z, w = rot.as_quat()
    return w, x, y, z

def main():
    # Load Orientation Data (.xyz) (Unchanged from your working version)
    try:
        print(f"Reading orientation data from {XYZ_FILE_PATH2}...")
        df_xyz = pd.read_csv(XYZ_FILE_PATH2, sep='\s+', comment='#', names=[
            'label', 'X0', 'Y0', 'Z0', 'omega', 'phi', 'kappa', 
            'c', 'x0', 'y0', 'a3', 'a4', 'a5', 'a6', 'rho0'
        ])
        xyz_lookup = df_xyz.set_index('label').to_dict('index')
    except Exception as e:
        print(f"Error: Could not read .xyz file correctly: {e}")
        return

    # Scan the Image Directory
    print(f"Scanning images in {IMAGE_DIR2}...")
    if not os.path.exists(IMAGE_DIR2): return

    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG')
    image_files = sorted([f for f in os.listdir(IMAGE_DIR2) if f.endswith(valid_extensions)])
    
    # --- CHANGE 1: Use the actual depth directory path (DISP_DIR2) for checks ---
    disp_dir_exists = os.path.exists(DISP_DIR2)

    # 3. Build the Data for the new CSV
    output_data = []

    # Get the relative folder name for the depth maps (e.g., 'depth_maps')
    DISP_FOLDER_NAME = os.path.basename(DISP_DIR2)

    for i, filename in enumerate(image_files):
        # ... (Orientation/Position lookup is unchanged and working) ...
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        tx, ty, tz = 0.0, 0.0, 0.0

        lookup_filename = filename
        if FILENAME_SUFFIX_TO_REMOVE in filename:
            name_part, ext_part = os.path.splitext(filename)
            # The XYZ file is matched by removing '_res'
            lookup_filename = name_part.replace(FILENAME_SUFFIX_TO_REMOVE, '') + ext_part

        if lookup_filename in xyz_lookup:
            data = xyz_lookup[lookup_filename]
            tx, ty, tz = data['X0'], data['Y0'], data['Z0']
            qw, qx, qy, qz = euler_to_quaternion(data['omega'], data['phi'], data['kappa'])
        
        # --- CRITICAL FIX: Generate the depth map filename correctly ---
        base_name_with_res = os.path.splitext(filename)[0]
        
        # 1. Remove the '_res' from the image base name: 
        # e.g., '...DxO_res' becomes '...DxO'
        base_name_clean = base_name_with_res
        if base_name_clean.endswith(FILENAME_SUFFIX_TO_REMOVE):
            base_name_clean = base_name_clean[:-len(FILENAME_SUFFIX_TO_REMOVE)]
            
        # 2. Append the correct DEPTH_SUFFIX: e.g., '...DxO' + '_depth_res.tiff'
        disp_filename = base_name_clean + DEPTH_SUFFIX
        
        # The full path to the depth file on the disk
        full_disp_path = os.path.join(DISP_DIR2, disp_filename)

        # --- CRITICAL CHANGE 3: Only process if the depth file exists ---
        if disp_dir_exists and os.path.exists(full_disp_path):
            # If it exists, set the RELATIVE path for the CSV
            
            # The relative path for the CSV (assuming base folder is Dataset-2)
            disp_path = os.path.join(DISP_FOLDER_NAME, disp_filename)
            
            # camera_l path is relative to its own folder (i.e., just the filename)
            camera_l_path = os.path.join('undistorted_images', filename)

            row = {
                'id': i,
                'camera_l': camera_l_path,
                'disp': disp_path,
                'qw': qw, 'qx': qy, 'qy': qy, 'qz': qz,
                'tx': tx, 'ty': ty, 'tz': tz
            }
            output_data.append(row)
        else:
            # Optional: Print a warning for files that are skipped
            print(f"Warning: Skipping {filename}. Corresponding depth map not found at {full_disp_path}")


    # 4. Create DataFrame and Save
    if output_data:
        df_out = pd.DataFrame(output_data)
        cols = ['id', 'camera_l', 'disp', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz']
        df_out = df_out[cols]
        # Ensure the output CSV is saved in the correct path relative to your script
        output_path = os.path.join(os.path.dirname(XYZ_FILE_PATH2), OUTPUT_CSV2)
        df_out.to_csv(output_path, index=False, sep='\t') 
        print(f"Success! Created {output_path} with {len(df_out)} entries.")
    else:
        print("No matching images with depth maps found.")

if __name__ == "__main__":
    main()