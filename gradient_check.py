import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import your model and dataloader
from m4depth_network import M4Depth
from dataloaders.midair import DataLoaderMidAir
from collections import namedtuple

def debug_gradient_flow():
    print("--- Testing Gradient Flow ---")
    
    # Configuration
    SEQ_LEN = 4
    DB_SEQ_LEN = 8
    BATCH_SIZE = 2  # Smaller batch for debugging
    RECORDS_PATH = "data/midair/train_data/"
    DB_PATH = "datasets/MidAir"

    # Setup settings
    DataloaderParameters = namedtuple('DataloaderParameters', 
                                      ('db_path_config', 'records_path', 'db_seq_len', 'seq_len', 'augment'))
    
    settings = DataloaderParameters(
        db_path_config={"midair": DB_PATH},
        records_path=RECORDS_PATH,
        db_seq_len=DB_SEQ_LEN,
        seq_len=SEQ_LEN,
        augment=False
    )

    # Initialize dataloader and get one batch
    loader = DataLoaderMidAir()
    loader.get_dataset("train", settings, batch_size=BATCH_SIZE)
    dataset = loader.dataset
    
    # Get one batch for testing
    sample = next(iter(dataset.take(1)))
    print(f"Sample keys: {list(sample.keys())}")
    print(f"new_traj shape: {sample['new_traj'].shape}")
    print(f"new_traj values: {sample['new_traj'].numpy()}")

    # Initialize model
    model = M4Depth(depth_type="map", nbre_levels=6, is_training=True)
    
    # Test forward pass
    print("\n--- Testing Forward Pass ---")
    
    # Prepare data in the format expected by the model
    seq_len = sample["depth"].get_shape().as_list()[1]
    traj_samples = [{} for i in range(seq_len)]
    attribute_list = ["depth", "RGB_im", "new_traj", "rot", "trans"]
    
    for key in attribute_list:
        value_list = tf.unstack(sample[key], axis=1)
        for i, item in enumerate(value_list):
            traj_samples[i][key] = item

    # Test call method directly
    print("Testing model call...")
    try:
        with tf.GradientTape() as tape:
            preds = model([traj_samples, sample["camera"]], training=True)
            print(f"Predictions type: {type(preds)}")
            print(f"Number of sequence predictions: {len(preds)}")
            
            # Check if we have actual depth estimates
            if preds and len(preds) > 0:
                last_seq_preds = preds[-1]
                print(f"Number of level predictions in last sequence: {len(last_seq_preds)}")
                
                for i, level_pred in enumerate(last_seq_preds):
                    print(f"Level {i} keys: {list(level_pred.keys())}")
                    for key, value in level_pred.items():
                        print(f"  {key} shape: {value.shape}")
            
            # Test loss computation
            gts = []
            for traj_sample in traj_samples:
                gts.append({
                    "depth": traj_sample["depth"], 
                    "parallax": depth2parallax(traj_sample["depth"], traj_sample["rot"], traj_sample["trans"], sample["camera"])
                })
            
            loss = model.m4depth_loss(gts, preds)
            print(f"Loss computed: {loss}")
            print(f"Loss is finite: {tf.math.is_finite(loss)}")
            
        # Test gradient computation
        print("\n--- Testing Gradient Computation ---")
        trainable_vars = model.trainable_variables
        print(f"Number of trainable variables: {len(trainable_vars)}")
        
        gradients = tape.gradient(loss, trainable_vars)
        non_none_grads = [g for g in gradients if g is not None]
        zero_grads = [g for g in non_none_grads if tf.reduce_sum(tf.abs(g)) == 0]
        
        print(f"Non-None gradients: {len(non_none_grads)}/{len(gradients)}")
        print(f"Zero gradients: {len(zero_grads)}/{len(non_none_grads)}")
        
        if non_none_grads:
            # Print first few gradients to see their values
            for i, grad in enumerate(non_none_grads[:5]):
                print(f"Gradient {i} - shape: {grad.shape}, norm: {tf.norm(grad).numpy()}")
        
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_gradient_flow()