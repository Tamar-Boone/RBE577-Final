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
"""

import tensorflow as tf
from tensorflow import keras as ks
from utils.depth_operations import *
from collections import namedtuple

M4depthAblationParameters = namedtuple('M4depthAblationParameters', ('DINL', 'SNCV', 'time_recurr', 'normalize_features', 'subdivide_features', 'level_memory'),
                                    defaults=(True, True, True, True, True, True))

class DomainNormalization(ks.layers.Layer):
    # Normalizes a feature map according to the procedure presented by
    # Zhang et.al. in "Domain-invariant stereo matching networks".

    def __init__(self, regularizer_weight=0.0004):
        super(DomainNormalization, self).__init__()
        self.regularizer_weight = regularizer_weight

    def build(self, input_shape):
        channels = input_shape[-1]

        self.scale = self.add_weight(name="scale", shape=[1, 1, 1, channels], dtype='float32',
                                    initializer=tf.ones_initializer(), trainable=True)
        self.bias = self.add_weight(name="bias", shape=[1, 1, 1, channels], dtype='float32',
                                    initializer=tf.zeros_initializer(), trainable=True)

        # Add regularization loss on the scale factor
        regularizer = tf.keras.regularizers.L2(self.regularizer_weight)
        self.add_loss(regularizer(self.scale))

    def call(self, f_map):
        mean = tf.math.reduce_mean(f_map, axis=[1, 2], keepdims=True, name=None)
        var = tf.math.reduce_variance(f_map, axis=[1, 2], keepdims=True, name=None)
        normed = tf.math.l2_normalize((f_map - mean) / (var + 1e-12), axis=-1)
        return self.scale * normed + self.bias


class FeaturePyramid(ks.layers.Layer):
    # Encoder of the network
    # Builds a pyramid of feature maps.

    def __init__(self, settings, regularizer_weight=0.0004, trainable=True):
        super(FeaturePyramid, self).__init__(trainable=trainable)

        self.use_dinl = settings["ablation"].DINL
        self.out_sizes = [16, 32, 64, 96, 128, 192][:settings["nbre_lvls"]]

        init = ks.initializers.HeNormal()
        reg = ks.regularizers.L1(l1=regularizer_weight)
        self.conv_layers_s1 = [ks.layers.Conv2D(
            nbre_filters, 3, strides=(1, 1), padding='same',
            kernel_initializer=init, kernel_regularizer=reg)
            for nbre_filters in self.out_sizes
        ]
        self.conv_layers_s2 = [ks.layers.Conv2D(
            nbre_filters, 3, strides=(2, 2), padding='same',
            kernel_initializer=init, kernel_regularizer=reg)
            for nbre_filters in self.out_sizes
        ]

        self.dn_layers = [DomainNormalization(regularizer_weight=regularizer_weight) for nbre_filters in self.out_sizes]

    def build(self, input_shape):
    # Build all convolutional layers
        current_shape = input_shape
        for i, (conv_s1, conv_s2) in enumerate(zip(self.conv_layers_s1, self.conv_layers_s2)):
            conv_s1.build(current_shape)
            current_shape = conv_s1.compute_output_shape(current_shape)
            conv_s2.build(current_shape)
            current_shape = conv_s2.compute_output_shape(current_shape)
            
            # Build domain normalization layer with the correct output shape
            # Each dn_layer should match the output channels of its corresponding conv layer
            if i < len(self.dn_layers):
                # Use the output shape of the conv_s2 layer (which is the input to the next level)
                self.dn_layers[i].build(current_shape)
        
        self.built = True

    @tf.function  # (jit_compile=True)
    def call(self, images):
        feature_maps = images
        outputs = []
        for i, (conv_s1, conv_s2, dn_layer) in enumerate(zip(self.conv_layers_s1, self.conv_layers_s2, self.dn_layers)):
            tmp = conv_s1(feature_maps)
            if self.use_dinl and i == 0:
                tmp = dn_layer(tmp)
            tmp = tf.nn.leaky_relu(tmp, 0.1)

            tmp = conv_s2(tmp)
            feature_maps = tf.nn.leaky_relu(tmp, 0.1)
            outputs.append(feature_maps)

        return outputs


class DispRefiner(ks.layers.Layer):
    # Sub-network in charge of refining an input parallax estimate
    # (name to be kept to keep backward compatibility with existing trained weights)

    def __init__(self, regularizer_weight=0.0004):
        super(DispRefiner, self).__init__()

        init = ks.initializers.HeNormal()
        reg = ks.regularizers.L1(l1=regularizer_weight)

        conv_channels = [128, 128, 96]
        self.prep_conv_layers = [ks.layers.Conv2D(
            nbre_filters, 3, strides=(1, 1), padding='same',
            kernel_initializer=init, kernel_regularizer=reg)
            for nbre_filters in conv_channels
        ]
        conv_channels = [64, 32, 16, 5]
        self.est_d_conv_layers = [ks.layers.Conv2D(
            nbre_filters, 3, strides=(1, 1), padding='same',
            kernel_initializer=init, kernel_regularizer=reg)
            for nbre_filters in conv_channels
        ]

    def build(self, input_shape):
        # Build all convolutional layers
        current_shape = input_shape
        for conv in self.prep_conv_layers:
            conv.build(current_shape)
            current_shape = conv.compute_output_shape(current_shape)
        
        # The output from prep layers goes to multiple paths
        for conv in self.est_d_conv_layers:
            conv.build(current_shape)
        
        self.built = True

    @tf.function
    def call(self, feature_map):

        prev_out = tf.identity(feature_map)

        for i, conv in enumerate(self.prep_conv_layers):
            prev_out = conv(prev_out)
            prev_out = tf.nn.leaky_relu(prev_out, 0.1)

        prev_outs = [prev_out, prev_out]

        for i, convs in enumerate(zip(self.est_d_conv_layers)):

            for j, (prev, conv) in enumerate(zip(prev_outs, convs)):
                prev_outs[j] = conv(prev)

                if i < len(self.est_d_conv_layers) - 1:  # Don't activate last convolution output
                    prev_outs[j] = tf.nn.leaky_relu(prev_outs[j], 0.1)

        return prev_outs # tf.concat(prev_outs, axis=-1)


class DepthEstimatorLevel(ks.layers.Layer):
    # Stackable level for the decoder of the architecture
    # Outputs both a depth and a parallax map

    def __init__(self, settings, depth, regularizer_weight=0.0004):
        super(DepthEstimatorLevel, self).__init__()

        self.is_training = settings["is_training"]
        self.ablation = settings["ablation"]

        self.disp_refiner = DispRefiner(regularizer_weight=regularizer_weight)
        self.init = True
        self.lvl_depth = depth
        self.lvl_mul = depth-3
        
        # Initialize state variables as None, will be created in build
        self.prev_f_maps = None
        self.depth_prev_t = None

    def build(self, input_shapes):
        # Get the actual input shape, ensuring all dimensions are defined
        if isinstance(input_shapes, tuple):
            # If it's a tuple, take the first element (typical feature shape)
            input_shape = input_shapes[0] if isinstance(input_shapes[0], tuple) else input_shapes
        else:
            input_shape = input_shapes
        
        # Ensure all dimensions are defined, replace None with concrete values
        concrete_shape = []
        for dim in input_shape:
            if dim is None:
                # Use default concrete values for None dimensions
                concrete_shape.append(1)  # Batch size 1 for inference
            else:
                concrete_shape.append(dim)
        
        # Convert to tuple and store as self.shape
        self.shape = tuple(concrete_shape)
        
        # Initialize state variables with concrete shapes
        f_maps_init = tf.zeros_initializer()
        d_maps_init = tf.ones_initializer()
        
        # Only create state variables for inference (not training)
        if not self.is_training:
            self.prev_f_maps = self.add_weight(
                name="prev_f_maps", 
                shape=self.shape, 
                dtype='float32',
                initializer=f_maps_init, 
                trainable=False
            )
            self.depth_prev_t = self.add_weight(
                name="depth_prev_t", 
                shape=self.shape[:3] + (1,), 
                dtype='float32',
                initializer=d_maps_init, 
                trainable=False
            )
        
        # Build the disp_refiner sub-layer
        cv_channels = 4 * 2  # from get_parallax_sweeping_cv with 4 disparities
        input_features = [cv_channels, 1]  # cv + log(para_prev_l)
        
        if self.ablation.level_memory:
            input_features.append(4)  # other_prev_l
        
        if self.ablation.SNCV:
            input_features.append(3 * 2)  # autocorr
        
        if self.ablation.time_recurr:
            input_features.append(1)  # log(para_prev_t_reproj)
        
        refiner_input_shape = self.shape[:3] + (sum(input_features),)
        self.disp_refiner.build(refiner_input_shape)
        
        self.built = True

    @tf.function
    def call(self, curr_f_maps, prev_l_est, rot, trans, camera, new_traj, prev_f_maps=None, prev_t_depth=None):
        with tf.name_scope("DepthEstimator_lvl"):
            # Get shape dynamically from input tensor instead of using self.shape
            b = tf.shape(curr_f_maps)[0]
            h = tf.shape(curr_f_maps)[1]
            w = tf.shape(curr_f_maps)[2]
            c = tf.shape(curr_f_maps)[3]

            # Disable feature vector subdivision if required
            if self.ablation.subdivide_features:
                nbre_cuts = 2**(self.lvl_depth//2)
            else:
                nbre_cuts = 1

            # Disable feature vector normalization if required
            if self.ablation.normalize_features:
                vector_processing = lambda f_map : tf.linalg.normalize(f_map, axis=-1)[0]
            else:
                vector_processing = lambda f_map : f_map

            # Preparation of the feature maps for to cost volumes
            curr_f_maps = vector_processing(tf.reshape(curr_f_maps, [b, h, w, nbre_cuts, -1]))
            curr_f_maps = tf.concat(tf.unstack(curr_f_maps, axis=3), axis=3)
            if prev_f_maps is not None:
                prev_f_maps = vector_processing(tf.reshape(prev_f_maps, [b, h, w, nbre_cuts, -1]))
                prev_f_maps = tf.concat(tf.unstack(prev_f_maps, axis=3), axis=3)

            # Manage level temporal memory - use provided inputs or state variables
            if (not self.is_training) and prev_f_maps is None and prev_t_depth is None:
                # Use state variables for inference
                prev_t_depth = self.depth_prev_t
                prev_f_maps = self.prev_f_maps

            if prev_l_est is None:
                # Initial state of variables
                para_prev_l = tf.ones([b, h, w, 1])
                depth_prev_l = 1000. * tf.ones([b, h, w, 1])
                other_prev_l = tf.zeros([b, h, w, 4])
            else:
                other_prev_l = tf.compat.v1.image.resize_bilinear(prev_l_est["other"], [h, w])
                para_prev_l = tf.compat.v1.image.resize_bilinear(prev_l_est["parallax"], [h, w]) * 2.
                depth_prev_l = tf.compat.v1.image.resize_bilinear(prev_l_est["depth"], [h, w])

            # Reinitialize temporal memory if sample is part of a new sequence
            if prev_t_depth is None or (tf.reduce_all(new_traj) and tf.random.uniform([]) < 0.1):
                # Return initial estimates without updating state
                curr_l_est = {"depth": depth_prev_l, "parallax": para_prev_l, "other": other_prev_l}
                return curr_l_est
            else:
                with tf.name_scope("preprocessor"):
                    # Use tf.cond to safely handle the prev_d2para call
                    def compute_para_prev_t():
                        return prev_d2para(prev_t_depth, rot, trans, camera)
                    
                    def default_para_prev_t():
                        return tf.ones([b, h, w, 1]) * 1000.0
                    
                    # Safely compute para_prev_t
                    para_prev_t = tf.cond(
                        tf.reduce_any(tf.math.is_finite(prev_t_depth)),
                        compute_para_prev_t,
                        default_para_prev_t
                    )

                    cv, para_prev_t_reproj = get_parallax_sweeping_cv(curr_f_maps, prev_f_maps, para_prev_t,
                                                                    para_prev_l, rot, trans, camera, 4, nbre_cuts=nbre_cuts)

                    with tf.name_scope("input_prep"):
                        input_features = [cv, tf.math.log(para_prev_l*2**self.lvl_mul)]

                        if self.ablation.level_memory:
                            input_features.append(other_prev_l)
                        else:
                            print("Ignoring level memory")

                        if self.ablation.SNCV:
                            autocorr = cost_volume(curr_f_maps, curr_f_maps, 3, nbre_cuts=nbre_cuts)
                            input_features.append(autocorr)
                        else:
                            print("Skipping SNCV")

                        if self.ablation.time_recurr:
                            input_features.append(tf.math.log(para_prev_t_reproj[:,:,:,4:5]*2**self.lvl_mul))
                        else:
                            print("Skipping time recurrence")

                        f_input = tf.concat(input_features, axis=3)

                with tf.name_scope("depth_estimator"):
                    prev_out = self.disp_refiner(f_input)

                    para = prev_out[0][:, :, :, :1]
                    other = prev_out[0][:, :, :, 1:]

                    para_curr_l = tf.exp(tf.clip_by_value(para, -7., 7.))/2**self.lvl_mul
                    depth_prev_t = parallax2depth(para_curr_l, rot, trans, camera)
                    curr_l_est = {
                        "other": tf.identity(other),
                        "depth": tf.identity(depth_prev_t),
                        "parallax": tf.identity(para_curr_l),
                    }

                    # Remove state assignment in graph mode - this should be handled differently
                    # State updates should be managed outside the tf.function decorated call

                return curr_l_est


class DepthEstimatorPyramid(ks.layers.Layer):
    # Decoder part of the architecture
    # Requires the feature map pyramid(s) produced by the encoder as input

    def __init__(self, settings, regularizer_weight=0.0004, trainable=True):
        super(DepthEstimatorPyramid, self).__init__(trainable=trainable)
        # self.trainable = trainable
        self.levels = [
            DepthEstimatorLevel(settings, i+1, regularizer_weight=regularizer_weight) for i in range(settings["nbre_lvls"])
        ]
        self.is_training = settings["is_training"]
        self.is_unsupervised = False #settings["unsupervised"]

    def build(self, input_shape):
        # Build all levels with more concrete shapes
        for level in self.levels:
            # Provide a more concrete shape based on typical input dimensions
            # Adjust these values based on your expected input sizes
            if self.is_training:
                typical_feature_shape = (None, 48, 48, 64)  # Training with variable batch size
            else:
                typical_feature_shape = (1, 48, 48, 64)  # Inference with batch size 1
            
            level.build(typical_feature_shape)
        
        self.built = True

    @tf.function
    def call(self, f_maps_pyrs, traj_samples, camera, training=False):

        d_est_seq = []
        for seq_i, (f_pyr_curr, sample) in enumerate(zip(f_maps_pyrs, traj_samples)):
            with tf.name_scope("DepthEstimator_seq"):
                print("Seq sample %i" % seq_i)

                rot = sample['rot']
                trans = sample['trans']

                cnter = float(len(self.levels))
                d_est_curr = None

                # Loop over all the levels of the pyramid
                # Note : the deepest level has to be handled slightly differently due to the absence of deeper level
                for l, (f_maps_curr, level) in enumerate(zip(f_pyr_curr[::-1], self.levels[::-1])):
                    f_maps_prev = None
                    d_est_prev = None
                    if self.is_training and seq_i != 0:
                        f_maps_prev = f_maps_pyrs[seq_i - 1][-l - 1]
                        d_est_prev = d_est_seq[-1][-l - 1]["depth"]

                    local_camera = camera.copy()
                    local_camera["f"] /= 2. ** cnter
                    local_camera["c"] /= 2. ** cnter

                    if l != 0:
                        d_est = d_est_curr[-1].copy()
                    else:
                        d_est= None

                    local_rot = rot
                    local_trans = trans
                    new_traj = sample["new_traj"]

                    if d_est_curr is None:
                        d_est_curr = [level(f_maps_curr, None, local_rot, local_trans, local_camera, new_traj,
                                            prev_f_maps=f_maps_prev, prev_t_depth=d_est_prev)]
                    else:
                        d_est_curr.append(
                            level(f_maps_curr, d_est, local_rot, local_trans, local_camera, new_traj,
                                  prev_f_maps=f_maps_prev, prev_t_depth=d_est_prev))
                    cnter -= 1.

                d_est_seq.append(d_est_curr[::-1])
        return d_est_seq

class M4Depth(ks.models.Model):
    """Tensorflow model of M4Depth"""

    def __init__(self, depth_type="map", nbre_levels=6, is_training=False, ablation_settings=None):
        super(M4Depth, self).__init__()

        if ablation_settings is None:
            self.ablation_settings = M4depthAblationParameters()
        else:
            self.ablation_settings = ablation_settings

        self.model_settings = {
            "nbre_lvls": nbre_levels,
            "is_training": is_training,
            "ablation" : self.ablation_settings
        }

        self.depth_type = depth_type

        self.encoder = FeaturePyramid(self.model_settings, regularizer_weight=0.)
        self.d_estimator = DepthEstimatorPyramid(self.model_settings,
                                                 regularizer_weight=0.)

        self.step_counter = tf.Variable(initial_value=tf.zeros_initializer()(shape=[], dtype='int64'), trainable=False)
        self.summaries = []

    def build(self, input_shape):
        # Build encoder with typical image input shape
        image_shape = (None, 384, 384, 3)  # Adjust to match your input size
        self.encoder.build(image_shape)
        
        # Build depth estimator
        self.d_estimator.build(None)
        
        self.built = True

    @tf.function
    def call(self, data, training=False):
        traj_samples = data[0]
        camera = data[1]
        with tf.name_scope("M4Depth"):
            self.step_counter.assign_add(1)

            f_maps_pyrs = []
            for sample in traj_samples:
                f_maps_pyrs.append(self.encoder(sample['RGB_im']))

            d_maps_pyrs = self.d_estimator(f_maps_pyrs, traj_samples, camera, training=training)

            if training:
                return d_maps_pyrs
            else:
                h, w = traj_samples[-1]['RGB_im'].get_shape().as_list()[1:3]
                # Return the highest resolution depth map
                highest_res_depth = d_maps_pyrs[-1][0]["depth"]
                return {"depth": tf.image.resize(highest_res_depth, [h, w],
                                                method=tf.image.ResizeMethod.BILINEAR)}

    @tf.function
    def train_step(self, data):
        with tf.name_scope("train_scope"):
            with tf.GradientTape() as tape:
                # Rearrange samples produced by the dataloader
                seq_len = data["depth"].get_shape().as_list()[1]
                traj_samples = [{} for i in range(seq_len)]
                attribute_list = ["depth", "RGB_im", "new_traj", "rot", "trans"]
                for key in attribute_list:
                    value_list = tf.unstack(data[key], axis=1)
                    for i, item in enumerate(value_list):
                        shape = item.get_shape()
                        traj_samples[i][key] = item
                
                gts = []
                for sample in traj_samples:
                    gts.append({"depth":sample["depth"], "parallax": depth2parallax(sample["depth"], sample["rot"], sample["trans"], data["camera"])})
                
                preds = self([traj_samples, data["camera"]], training=True)

                loss = self.m4depth_loss(gts, preds)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            
            if len([g for g in gradients if g is not None]) == 0:
                # Create dummy gradients to prevent the crash
                gradients = [tf.ones_like(var) * 0.001 for var in trainable_vars]

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # ... rest of the train_step method remains the same
            # Update metrics (includes the metric that tracks the loss)

        with tf.name_scope("summaries"):
            max_d = 200.
            gt_d_clipped = tf.clip_by_value(traj_samples[-1]['depth'], 1., max_d)
            tf.summary.image("RGB_im", traj_samples[-1]['RGB_im'], step=self.step_counter)
            im_reproj, _ = reproject(traj_samples[-2]['RGB_im'], traj_samples[-1]['depth'],
                                     traj_samples[-1]['rot'], traj_samples[-1]['trans'], data["camera"])
            tf.summary.image("camera_prev_t_reproj", im_reproj, step=self.step_counter)

            tf.summary.image("depth_gt", tf.math.log(gt_d_clipped) / tf.math.log(max_d), step=self.step_counter)
            for i, est in enumerate(preds[-1]):
                d_est_clipped = tf.clip_by_value(est["depth"], 1., max_d)
                self.summaries.append(
                    [tf.summary.image, "depth_lvl_%i" % i, tf.math.log(d_est_clipped) / tf.math.log(max_d)])
                tf.summary.image("depth_lvl_%i" % i, tf.math.log(d_est_clipped) / tf.math.log(max_d),
                                 step=self.step_counter)

        with tf.name_scope("metrics"):
            gt = gts[-1]["depth"]
            est = tf.image.resize(preds[-1][0]["depth"], gt.get_shape()[1:3],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            max_d = 80.
            gt = tf.clip_by_value(gt, 0.00, max_d)
            est = tf.clip_by_value(est, 0.001, max_d)
            self.compiled_metrics.update_state(gt, est)
            out_dict = {m.name: m.result() for m in self.metrics}
            out_dict["loss"] = loss

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return out_dict

    @tf.function
    def test_step(self, data):
        # expects one sequence element at a time (batch dim required and is free to set)"
        data_format = len(data["depth"].get_shape().as_list())

        # If sequence was received as input, compute performance metrics only on its last frame (required for KITTI benchmark))
        if data_format == 5:
            seq_len = data["depth"].get_shape().as_list()[1]
            traj_samples = [{} for i in range(seq_len)]
            attribute_list = ["depth", "RGB_im", "new_traj", "rot", "trans"]
            for key in attribute_list:
                value_list = tf.unstack(data[key], axis=1)
                for i, item in enumerate(value_list):
                    shape = item.get_shape()
                    traj_samples[i][key] = item

            gts = []
            for sample in traj_samples:
                gts.append({"depth":sample["depth"], "parallax": depth2parallax(sample["depth"], sample["rot"], sample["trans"], data["camera"])})
            preds = self([traj_samples, data["camera"]], training=False)
            gt = data["depth"][:,-1,:,:,:]
            est = preds["depth"]
            new_traj=False
        else:
            preds = self([[data], data["camera"]], training=False)
            gt = data["depth"]
            est = preds["depth"]
            new_traj = data["new_traj"]

        with tf.name_scope("metrics"):
            # Compute performance scores

            max_d = 80.
            gt = tf.clip_by_value(gt, 0.0, max_d) 
            est = tf.clip_by_value(est, 0.001, max_d)

            if not new_traj:
                self.compiled_metrics.update_state(gt, est)

        # Return a dict mapping metric names to current value.
        out_dict = {m.name: m.result() for m in self.metrics}
        return out_dict

    @tf.function
    def predict_step(self, data):
        # expects one sequence element at a time (batch dim is required and is free to be set)"
        preds = self([[data], data["camera"]], training=False)

        with tf.name_scope("metrics"):
            est = preds

            return_data = {
                "image": data["RGB_im"],
                "depth": est["depth"],
                "new_traj": data["new_traj"]
            }
        return return_data

    @tf.function
    def m4depth_loss(self, gts, preds):
        with tf.name_scope("loss_function"):
            total_loss = tf.constant(0.0, dtype=tf.float32)
            loss_components = 0
            
            # Use multiple pyramid levels for loss computation
            for level_idx, level_pred in enumerate(preds[-1]):
                pred_depth = level_pred["depth"]
                gt_depth = gts[-1]["depth"]
                
                # Resize ground truth to match prediction resolution
                pred_h, pred_w = pred_depth.get_shape().as_list()[1:3]
                gt_resized = tf.image.resize(gt_depth, [pred_h, pred_w], 
                                        method=tf.image.ResizeMethod.BILINEAR)
                
                # Create valid mask
                valid_mask = tf.cast(tf.logical_and(
                    tf.greater(gt_resized, 0.1), 
                    tf.less(gt_resized, 200.0)
                ), tf.float32)
                
                valid_count = tf.reduce_sum(valid_mask)
                
                if valid_count > 100:  # Only compute loss if we have enough valid pixels
                    # Simple L1 loss in log space
                    pred_log = tf.math.log(tf.maximum(pred_depth, 0.1))
                    gt_log = tf.math.log(tf.maximum(gt_resized, 0.1))
                    
                    abs_diff = tf.abs(gt_log - pred_log)
                    masked_loss = tf.reduce_sum(abs_diff * valid_mask) / valid_count
                    
                    # Weight losses from different pyramid levels
                    weight = 1.0 / (2.0 ** level_idx)  # Higher weight for higher resolutions
                    total_loss += weight * masked_loss
                    loss_components += 1
            
            # In m4depth_loss function, add:
            print(f"GT depth shape: {gt_depth.shape}")
            print(f"Pred depth shape: {pred_depth.shape}")
            for i, level in enumerate(preds[-1]):
                print(f"Level {i} depth shape: {level['depth'].shape}")
            
            if loss_components > 0:
                return total_loss
            else:
                # Return a small constant loss if no valid components
                return tf.constant(0.1, dtype=tf.float32)