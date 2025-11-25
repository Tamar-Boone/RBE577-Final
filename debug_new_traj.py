import tensorflow as tf
import os
import argparse
from collections import namedtuple

# Import your dataloader
from dataloaders.midair import DataLoaderMidAir

# Define the settings tuple expected by the dataloader
DataloaderParameters = namedtuple('DataloaderParameters', 
                                  ('db_path_config', 'records_path', 'db_seq_len', 'seq_len', 'augment'))

def debug_dataloader():
    print("--- Starting Dataloader Debug ---")
    
    # 1. Configuration (Matching your script)
    SEQ_LEN = 4
    DB_SEQ_LEN = 8
    BATCH_SIZE = 3
    RECORDS_PATH = "data/midair/train_data/"
    
    # Path to images (db_path). Adjust if your images are elsewhere, 
    # but for checking 'new_traj' (which comes from CSVs in records_path), this acts as a placeholder.
    DB_PATH = "datasets/MidAir"

    # 2. Setup Settings
    settings = DataloaderParameters(
        db_path_config={"midair": DB_PATH},
        records_path=RECORDS_PATH,
        db_seq_len=DB_SEQ_LEN,
        seq_len=SEQ_LEN,
        augment=False # Disable augmentation to keep things simple
    )

    # 3. Initialize Dataloader
    loader = DataLoaderMidAir()
    
    # 4. Build Dataset
    print(f"Building dataset with seq_len={SEQ_LEN}, batch_size={BATCH_SIZE}...")
    try:
        loader.get_dataset("train", settings, batch_size=BATCH_SIZE)
    except Exception as e:
        print(f"Error building dataset: {e}")
        return

    # 5. Inspect the first few batches
    print("\n--- Inspecting Batches ---")
    dataset = loader.dataset
    
    # Iterate through 3 batches
    for i, sample in enumerate(dataset.take(3)):
        new_traj = sample["new_traj"].numpy()
        ids = sample["trans"].numpy() # Just to see if we have data, usually 'id' isn't kept unless requested
        
        print(f"\nBatch {i}:")
        print(f"new_traj shape: {new_traj.shape} (Expected: [{BATCH_SIZE}, {SEQ_LEN}])")
        print("new_traj values (True=Start, False=Continuation):")
        print(new_traj)
        
        # Check if we have the expected pattern: [True, False, False, False]
        # Note: Depending on 'offset' in _cut_sequence, it might shift.
        # But we MUST see 'False' values for gradients to exist.
        
        all_true = new_traj.all()
        print(f"Are ALL values True? {all_true}")
        if all_true:
            print("❌ PROBLEM FOUND: All frames are marked as new trajectories. Model will skip training.")
        else:
            print("✅ OK: Found False values. Sequence continuation detected.")

if __name__ == "__main__":
    # Ensure TF doesn't hide errors
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    debug_dataloader()