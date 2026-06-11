import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, activations, Model


from fast_qrs_detector import qrs_detector, print_signal_with_qrs

################################## GSA BLOCK ###################################
class GSA_Block(layers.Layer):
    def __init__(self, K, **kwargs):
        '''
        K refers to the number of input channels for the GSA block
        '''
        super(GSA_Block, self).__init__(**kwargs)
        self.K = K
        self.max_pool = layers.MaxPool1D(pool_size=2, strides=2)
        self.upsample = layers.UpSampling1D(size=2)
        self.concat = layers.Concatenate(axis=-1)       # concat on the channel dimension (B, L, K)

        self.enc_conv1 = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')        # preparing for downsampling (top to middle)
        self.conv_mid1 = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')        # passing through the middle layer - red arrow
        
        self.enc_conv2 = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')        # preparing for downsampling (middle to bottom)
        self.bottleneck_1 = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')     # passing thorugh the bottom layer - red arrow 1
        self.bottleneck_2 = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')     # red arrow 2

        self.dec_conv2 = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')        # after upsampling (bottom to middle)
        self.dec_conv1 = layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')        # after upsampling (middle to top)


        self.final_conv = layers.Conv1D(filters=1, kernel_size=3, padding='same')

    def call(self, conv_input):
        middle_layer = self.enc_conv1(conv_input)
        middle_layer = self.max_pool(middle_layer)
        middle_layer = self.conv_mid1(middle_layer)

        bottom_layer = self.enc_conv2(middle_layer)
        bottom_layer = self.max_pool(bottom_layer)               

        bottom_layer = self.bottleneck_1(bottom_layer)           
        bottom_layer = self.bottleneck_2(bottom_layer)           

        bottom_layer_up = self.upsample(bottom_layer)              
        # before a concatentation operation, we need to ensure tensors are the
        # same length
        pad_len_mid = tf.shape(middle_layer)[1] - tf.shape(bottom_layer_up)[1]         
        bottom_layer_up = tf.pad(bottom_layer_up, [[0, 0], [0, pad_len_mid], [0, 0]])
        # ready to concatenate now
        middle_cat = self.concat([bottom_layer_up, middle_layer])        
        middle_layer = self.dec_conv2(middle_cat)      


        middle_layer_up = self.upsample(middle_layer)
        # before a concatentation operation, we need to ensure tensors are the
        # same length
        pad_len_top = tf.shape(conv_input)[1] - tf.shape(middle_layer_up)[1]
        middle_layer_up = tf.pad(middle_layer_up, [[0, 0], [0, pad_len_top], [0, 0]])
        # ready to concatenate now
        top_cat = self.concat([middle_layer_up, conv_input])        
        top_layer = self.dec_conv1(top_cat)      

        attention_weights_no_sig = self.final_conv(top_layer)
        attention_weights = tf.nn.sigmoid(attention_weights_no_sig)
        final_output = conv_input * (attention_weights + 1.0)

        return final_output, attention_weights, attention_weights_no_sig

def gsa_block_smoke_test():
    print("Initializing isolated GSA Block dry-run...")
        
    # 1. Setup mock dimensions (Batch=4, Length=500, Channels=32)
    B, L, K = 4, 500, 32
    
    # 2. Generate pseudo-random ECG feature maps simulating backbone inputs
    mock_backbone_features = tf.random.normal(shape=(B, L, K))
    print(f"-> Formed Input Tensor Shape: {mock_backbone_features.shape}")

    # 3. Instantiate the isolated block
    gsa_tester = GSA_Block(K)

    # 4. Pass the mock input data through the isolated block
    # Note: Keras dynamically builds weights upon this first call
    transformed_features, spatial_attention, _ = gsa_tester(mock_backbone_features)

    # 5. Diagnostic Dimension Checks
    print("\n--- RESULTS ---")
    print(f"Transformed Feature Map Output Shape : {transformed_features.shape}")
    print(f"Generated Attention Map Output Shape : {spatial_attention.shape}")

    # 6. Sanity Verifications
    # The feature output shape MUST perfectly equal the input shape
    assert transformed_features.shape == mock_backbone_features.shape, "Error: Feature maps morphed shape!"
    # The attention map MUST have squeezed the channels down to exactly 1
    assert spatial_attention.shape == (B, L, 1), "Error: Attention channel depth is not 1!"
    
    print("\n[SUCCESS]: GSA Block compiled, downsampled, upsampled, and executed without error.")

############################### GSA Architecture ###############################


class GSA(Model):
    def __init__(self):
        super(GSA, self).__init__()

        self.max_pool = layers.MaxPool1D(pool_size=2, strides=2)

        self.conv1 = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')
        self.gsa1 = GSA_Block(32)
        self.conv2 = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')
        self.gsa2 = GSA_Block(32)
        self.conv3 = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.gsa3 = GSA_Block(64)
        self.conv4 = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.gsa4 = GSA_Block(64)
        self.conv5 = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
        self.gsa5 = GSA_Block(128)

        self.gap = layers.GlobalAveragePooling1D()
        self.dense = layers.Dense(units=4, activation="sigmoid")


    def call(self, ecg_input):
        attention_weights_no_sigmoid = [ None ] * 5
        attention_weights = [ None ] * 5

        x = self.conv1(ecg_input)
        x, attention_weights[0], attention_weights_no_sigmoid[0] = self.gsa1(x)
        x = self.max_pool(x)
        
        x = self.conv2(x)
        x, attention_weights[1], attention_weights_no_sigmoid[1] = self.gsa2(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x, attention_weights[2], attention_weights_no_sigmoid[2] = self.gsa3(x)
        x = self.max_pool(x)

        x = self.conv4(x)
        x, attention_weights[3], attention_weights_no_sigmoid[3] = self.gsa4(x)
        x = self.max_pool(x)

        x = self.conv5(x)
        x, attention_weights[4], attention_weights_no_sigmoid[4] = self.gsa5(x)

    
        x = self.gap(x)
        classification_output = self.dense(x)

        return classification_output, attention_weights, attention_weights_no_sigmoid

        # Need help with the interpolation and getting the final attention outputs 

    
    def classification_loss(self, y_true_labels, y_pred_probs):
        """
        y_true_labels: Binary matrix of patient diagnoses (Shape: Batch, 30)
        y_pred_probs: Output from your dense layer (Shape: Batch, 30)
        """
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        return bce(y_true_labels, y_pred_probs)
    
    def attention_loss(self, y_true, attention_logits_list):
        """
        y_true: The true clinical target binary mask (Shape: Batch, 500, 1)
        attention_logits_list: Your collected `attention_weights_no_sigmoid` list of 5 tensors
        """
        total_dice_loss = 0.0
        target_length = tf.shape(y_true)[1]  # Safely extracts original length (e.g., 500)
        smooth = 1 

        for logits in attention_logits_list:
            probs = tf.nn.sigmoid(logits)
            
            probs = tf.expand_dims(probs, axis=-1)  # (B, L, 1, 1)
            rescaled = tf.image.resize(probs, size=[target_length, 1])
            rescaled_probs = tf.squeeze(rescaled, axis=-1)  # (B, target_length, 1)
            
            
            intersection = tf.reduce_sum(y_true * rescaled_probs, axis=1)
            denominator = tf.reduce_sum(y_true, axis=1) + tf.reduce_sum(rescaled_probs, axis=1)
            
            dice_coef = (2.0 * intersection + smooth) / (denominator + smooth)
            
            total_dice_loss += (1.0 - tf.reduce_mean(dice_coef))
            
        return total_dice_loss / len(attention_logits_list)

    
    def total_loss(self, true_labels, true_attention, pred_probs, attention_logits_list, alpha):
        if "NORM" in true_labels:
            return self.classification_loss(true_labels, pred_probs)
        else:
            loss_att = self.attention_loss(true_attention, attention_logits_list)
            
        loss_class = self.classification_loss(true_labels, pred_probs)

        return alpha * loss_class + (1 - alpha) * loss_att

############################ Guided Attention Maps #############################
QRS_DURATION = 0.1
QR_INTERVAL = 0.04
PR_INTERVAL = 0.3

class Attention_Maps():
    def generate_qrs_mask(self, ecg_signal, sampling_freq):
        num_samples = ecg_signal.shape[0]
        mask = np.zeros((num_samples, 1))

        qrs_results = qrs_detector(ecg_signal, sampling_freq)

        HALF_SAMPLES = int(sampling_freq * (QRS_DURATION / 2))
        for peak in qrs_results:
            start_idx = max(0, peak - HALF_SAMPLES)
            end_idx = min(num_samples, peak + HALF_SAMPLES)

            mask[start_idx:end_idx, 0] = 1

        return mask
    
    def generate_pr_mask(self, ecg_signal, sampling_freq):
        num_samples = ecg_signal.shape[0]
        mask = np.zeros((num_samples, 1))

        qrs_results = qrs_detector(ecg_signal, sampling_freq)


        for peak in qrs_results:
            end_idx = max(0, peak - int( sampling_freq * QR_INTERVAL ))
            start_idx = max(0, peak - int(sampling_freq * PR_INTERVAL))

            mask[start_idx:end_idx, 0] = 1

        return mask
        
    








################################## SMOKE TESTS ##################################

def gsa_architecture_smoke_test():
    print("==================================================")
    print("STARTING MACRO-ARCHITECTURE SMOKE CHECK...")
    print("==================================================\n")

    # 1. Define mock dataset dimensions
    BATCH_SIZE = 4
    SIGNAL_LENGTH = 500  # The L timeline dimension
    CHANNELS = 12         # e.g., Single-lead ECG input
    NUM_CLASSES = 4     # Your 4 diagnostic conditions

    print(f"[STEP 1]: Creating mock ECG batch data...")
    print(f"-> Target Batch Size        : {BATCH_SIZE}")
    print(f"-> Target Sequence Length  : {SIGNAL_LENGTH}")
    print(f"-> Input Channel Depth     : {CHANNELS}")
    
    # Generate pseudo-random ECG waves (simulating raw voltage readings)
    mock_ecg_inputs = tf.random.normal(shape=(BATCH_SIZE, SIGNAL_LENGTH, CHANNELS))
    print(f"[SUCCESS]: Generated Input Shape: {mock_ecg_inputs.shape}\n")

    # 2. Instantiate your complete GSA Model Network
    print(f"[STEP 2]: Initializing the GSA Model wrapper...")
    try:
        model = GSA()
        print("[SUCCESS]: GSA class initialized without errors.\n")
    except Exception as e:
        print(f"[CRITICAL FAILURE]: Could not instantiate GSA class. Error: {e}")
        return

    # 3. Perform a forward pass dry-run
    print(f"[STEP 3]: Pushing mock data forward through the pipeline...")
    try:
        # Keras will dynamically construct the internal weights on this first call
        cls_output, att_weights, att_logits = model(mock_ecg_inputs)
        print("[SUCCESS]: Forward pass completed without execution crashes.\n")
    except Exception as e:
        print(f"[CRITICAL FAILURE]: Model crashed during forward pass! Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Diagnostic Dimension Assertions
    print(f"[STEP 4]: Executing diagnostic shape validations...")
    print("--- OUTPUT DIMENSIONS ---")
    print(f"Diagnostic Classifications Shape : {cls_output.shape}")
    print(f"Total Attention Weight Arrays    : {len(att_weights)} tiers tracked")
    print(f"Total Attention Logit Arrays     : {len(att_logits)} tiers tracked")
    print("-------------------------")

    # Verification A: Classification Head Check
    # The output MUST be a 2D matrix of shape (Batch, Num_Classes)
    expected_cls_shape = (BATCH_SIZE, NUM_CLASSES)
    assert cls_output.shape == expected_cls_shape, \
        f"Shape Mismatch! Classification output is {cls_output.shape}, expected {expected_cls_shape}."
    print("-> [PASSED]: Classification matrix is correctly flattened and sized.")

    # Verification B: Structural Deep Tracking Check
    # Ensure all 5 GSA layers actually caught data
    assert len(att_weights) == 5 and len(att_logits) == 5, \
        f"Tracking Mismatch! Expected 5 layers, but got {len(att_weights)}."
    print("-> [PASSED]: Attention outputs successfully harvested from all 5 stacked GSA blocks.")

    # Verification C: Signal Property Check
    # Ensure final diagnostic predictions are valid probabilities [0, 1] due to sigmoid activation
    min_val = np.min(cls_output.numpy())
    max_val = np.max(cls_output.numpy())
    assert min_val >= 0.0 and max_val <= 1.0, \
        f"Activation Error! Output values range from {min_val} to {max_val}. Did you forget Sigmoid?"
    print("-> [PASSED]: Classification outputs are successfully bound between [0.0, 1.0].")


    print("[STEP 5]: Testing loss function...")
    mock_labels = tf.random.uniform((BATCH_SIZE, NUM_CLASSES))
    mock_attention = tf.random.uniform((BATCH_SIZE, SIGNAL_LENGTH, 1))
    loss_cls = model.classification_loss(mock_labels, cls_output)
    loss_att = model.attention_loss(mock_attention, att_logits)

    loss = model.total_loss(mock_labels, mock_attention, cls_output, att_logits, alpha=0.6)
    print(f"-> Classification Loss Value: {loss_cls.numpy()}")
    print(f"-> Attention Loss Value: {loss_att.numpy()}")

    print(f"-> Total Loss Value: {loss.numpy()}")


    print("\n==================================================")
    print("[GLOBAL SUCCESS]: GSA Class passed all sanity checks!")
    print("The macro-architecture is structurally sound and ready for real data.")
    print("==================================================")


if __name__ == "__main__":
    gsa_block_smoke_test()
    print()
    gsa_architecture_smoke_test()

