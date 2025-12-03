"""
----------------------------------------------------------------------------------------
Copyright (c) 2022 - Michael Fonder, University of Liège (ULiège), Belgium.

This program is free software: you can redistribute it and/or modify it under the terms
of the GNU Affero General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License along with this
program. If not, see < [ https://www.gnu.org/licenses/ | https://www.gnu.org/licenses/ ] >.
----------------------------------------------------------------------------------------

To run program
./scripts/1a-train-midair.sh pretrained_weights/midair

"""

# DISABLE XLA COMPLETELY AND AGGRESSIVELY
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_DISABLE_XLA'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Now import other modules
import argparse
from m4depth_options import M4DepthOptions

cmdline = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
model_opts = M4DepthOptions(cmdline)
cmd, test_args = cmdline.parse_known_args()

import tensorflow as tf
# Force disable XLA/JIT compilation
tf.config.optimizer.set_jit(False)
tf.config.optimizer.set_experimental_options({
    "constant_folding": True,
    "shape_optimization": True,
    "remapping": True,
    "arithmetic_optimization": True,
    "dependency_optimization": True,
    "loop_optimization": True,
    "function_optimization": True,
    "debug_stripper": True,
    "scoped_allocator_optimization": True,
    "implementation_selector": True,
    "auto_mixed_precision": False,
    "pin_to_host_optimization": True,
    "layout_optimizer": True
})
tf.config.set_soft_device_placement(True)

from tensorflow import keras
import numpy as np
import dataloaders as dl
from callbacks import *
from m4depth_network import *
from metrics import *
import time

if __name__ == '__main__':

    cmdline = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    model_opts = M4DepthOptions(cmdline)
    cmd, test_args = cmdline.parse_known_args()

    # configure tensorflow gpus
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

    enable_validation = cmd.enable_validation
    try:
        if cmd.mode == "validation":
            print('limit memory')
            tf.config.set_logical_device_configuration(physical_devices[0],
                                                       [tf.config.LogicalDeviceConfiguration(memory_limit=1200)])
    except:
        print("GPUs initialization failed")
        enable_validation = False
        pass

    working_dir = os.getcwd()
    print("The current working directory is : %s" % working_dir)

    chosen_dataloader = dl.get_loader(cmd.dataset)

    seq_len = cmd.seq_len
    nbre_levels = cmd.arch_depth
    ckpt_dir = cmd.ckpt_dir

    if cmd.mode == 'train' or cmd.mode == 'finetune':

        print("Training on %s" % cmd.dataset)
        tf.random.set_seed(42)
        chosen_dataloader.get_dataset("train", model_opts.dataloader_settings, batch_size=cmd.batch_size)
        data = chosen_dataloader.dataset

        print("=== DATA INSPECTION ===")
        print("Dataset sample inspection:")
        try:
            sample_batch = next(iter(data.take(1)))
            print("Batch keys:", sample_batch.keys())
            print("RGB shape:", sample_batch["RGB_im"].shape)
            print("Depth shape:", sample_batch["depth"].shape)
            print("new_traj shape:", sample_batch["new_traj"].shape)
            if "new_traj" in sample_batch:
                print("new_traj shape:", sample_batch["new_traj"].shape)
                # print("new_traj sample:", sample_batch["new_traj"].numpy())
        except Exception as e:
            print(f"Error inspecting data: {e}")
        print("=======================")

        model = M4Depth(depth_type=chosen_dataloader.depth_type,
                        nbre_levels=nbre_levels,
                        ablation_settings=model_opts.ablation_settings,
                        is_training=True)

        model_checkpoint_cbk = CustomCheckpointCallback(os.path.join(ckpt_dir,"train"), resume_training=True)

        # Use standard learning rate
        opt = tf.keras.optimizers.AdamW(
            learning_rate=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=1.0,
        )

        model.compile(optimizer=opt, metrics=[RootMeanSquaredLogError()])

        if enable_validation:
            val_cbk = [CustomKittiValidationCallback(cmd, args=test_args)]
        else:
            val_cbk = []
        
        gradient_monitor = GradientMonitor(log_frequency=10)

        if cmd.mode == 'finetune':
            nbre_epochs = model_checkpoint_cbk.resume_epoch + (20000 // chosen_dataloader.length)
        else:
            nbre_epochs = (220000 // chosen_dataloader.length)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            cooldown=5,
            verbose=1
        )

        # Standard training loop
        model.fit(data, epochs=nbre_epochs + 1,
                  initial_epoch=model_checkpoint_cbk.resume_epoch,
                  callbacks=[model_checkpoint_cbk, reduce_lr, gradient_monitor] + val_cbk)

    elif cmd.mode == 'eval' or cmd.mode == 'validation':

        if cmd.mode=="validation":
            weights_dir = os.path.join(ckpt_dir,"train")
        else:
            weights_dir = os.path.join(ckpt_dir,"best")

        print("Evaluating on %s" % cmd.dataset)
        chosen_dataloader.get_dataset("eval", model_opts.dataloader_settings, batch_size=1)
        data = chosen_dataloader.dataset

        model = M4Depth(nbre_levels=nbre_levels, ablation_settings=model_opts.ablation_settings)

        model_checkpoint_cbk = CustomCheckpointCallback(weights_dir, resume_training=True)
        model.compile(metrics=[AbsRelError(),
                               SqRelError(),
                               RootMeanSquaredError(),
                               RootMeanSquaredLogError(),
                               ThresholdRelError(1), ThresholdRelError(2), ThresholdRelError(3)])

        metrics = model.evaluate(data, callbacks=[model_checkpoint_cbk])

        if cmd.mode == 'validation':
            manager = BestCheckpointManager(os.path.join(ckpt_dir,"train"), os.path.join(ckpt_dir,"best"), keep_top_n=cmd.keep_top_n)
            perfs = {"abs_rel": [metrics[0]], "sq_rel": [metrics[1]], "rmse": [metrics[2]], "rmsel": [metrics[3]],
                     "a1": [metrics[4]], "a2": [metrics[5]], "a3": [metrics[6]]}
            manager.update_backup(perfs)
            string = ''
            for perf in metrics:
                string += format(perf, '.4f') + "\t\t"
            with open(os.path.join(*[ckpt_dir, "validation-perfs.txt"]), 'a') as file:
                file.write(string + '\n')
        else:
            np.savetxt(os.path.join(*[ckpt_dir, "perfs-" + cmd.dataset + ".txt"]), metrics, fmt='%.18e', delimiter='\t',
                       newline='\n')

    elif cmd.mode == "predict":
        chosen_dataloader.get_dataset("predict", model_opts.dataloader_settings, batch_size=1)
        data = chosen_dataloader.dataset

        model = M4Depth(nbre_levels=nbre_levels, ablation_settings=model_opts.ablation_settings)
        model.compile()
        model_checkpoint_cbk = CustomCheckpointCallback(os.path.join(ckpt_dir, "best"), resume_training=True)
        first_sample = data.take(1)
        model.predict(first_sample, callbacks=[model_checkpoint_cbk])

        is_first_run = True

        for i, sample in enumerate(data):
            if not is_first_run and sample["new_traj"]:
                print("End of trajectory")

            is_first_run = False

            est = model([[sample], sample["camera"]]) 
            d_est = est["depth"][0, :, :, :]       
            d_gt = sample['depth'][0, :, :, :]     
            i_rgb = sample['RGB_im'][0, :, :, :]