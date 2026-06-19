import os
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import sys

from src.train import (
    preprocess_signal, load_single_record, load_best_model,
    get_attention_mask, SIGNAL_LENGTH, TARGET_FS,
    LEAD_II_INDEX, CLASS_MAP
)

CLASS_NAMES = ['NORM', 'LBBB', 'RBBB', '1dAVB']




def predict_single_record(model, database_path, filename, true_label):
    raw_signal, fs = load_single_record(database_path, filename)
    processed = preprocess_signal(raw_signal, fs)
    
    # Add batch dimension
    signal_batch = tf.expand_dims(processed, axis=0)  # (1, 1200, 12)
    
    pred_probs, att_weights, att_logits = model(signal_batch, training=False)
    
    pred_class = tf.argmax(pred_probs, axis=1).numpy()[0]
    pred_prob  = pred_probs.numpy()[0]
    
    class_names = ['NORM', 'LBBB', 'RBBB', '1dAVB']
    print(f"True label     : {class_names[true_label]}")
    print(f"Predicted label: {class_names[pred_class]}")
    print(f"Probabilities  : ", {class_names[i]: f"{pred_prob[i]:.3f}" for i in range(4)})
    
    return pred_probs, att_weights, att_logits, processed


def visualise_attention(model, database_path, filename, true_label, fold_idx):
    raw_signal, fs = load_single_record(database_path, filename)
    processed = preprocess_signal(raw_signal, fs)
    signal_batch = tf.expand_dims(processed, axis=0)
    
    pred_probs, att_weights, att_logits, = model(signal_batch, training=False)
    
    # Get lead II from processed signal
    lead_ii = processed[:, LEAD_II_INDEX]
    
    # Generate ground truth mask for comparison
    gt_mask = get_attention_mask(lead_ii, true_label)
    
    # Time axis in seconds
    time_axis = np.arange(SIGNAL_LENGTH) / TARGET_FS
    
    class_names = ['NORM', 'LBBB', 'RBBB', '1dAVB']
    
    # att_weights is a list of 5 tensors, each (1, L_i, 1)
    # where L_i decreases with depth due to max pooling
    num_blocks = len(att_weights)
    
    fig = plt.figure(figsize=(15, 3 * (num_blocks + 2)))
    gs  = gridspec.GridSpec(num_blocks + 2, 1)
    
    # Plot lead II signal
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(time_axis, lead_ii, color='black', linewidth=0.8)
    ax0.set_title(f"Lead II — True: {class_names[true_label]}, "
                  f"Predicted: {class_names[tf.argmax(pred_probs[0]).numpy()]}")
    ax0.set_ylabel("Amplitude (mV)")
    ax0.set_xlabel("Time (s)")

    # Plot ground truth mask if it exists
    ax1 = fig.add_subplot(gs[1])
    if gt_mask is not None:
        ax1.fill_between(time_axis, gt_mask[:, 0], alpha=0.7, color='green')
        ax1.set_title("Ground Truth Attention Mask")
    else:
        ax1.text(0.5, 0.5, 'No attention mask (NORM)',
                 ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title("Ground Truth Attention Mask")
    ax1.set_ylabel("Mask")
    ax1.set_ylim(-0.1, 1.1)

    # Plot each GSA block's attention weights
    for block_idx in range(num_blocks):
        ax = fig.add_subplot(gs[block_idx + 2])
        
        # att_weights[block_idx] shape: (1, L_i, 1)
        block_weights = att_weights[block_idx][0, :, 0].numpy()
        block_length  = len(block_weights)
        
        # Interpolate back to full signal length for visual alignment
        block_time = np.linspace(0, SIGNAL_LENGTH / TARGET_FS, block_length)
        
        ax.plot(block_time, block_weights, color='red', linewidth=1.0)
        ax.fill_between(block_time, block_weights, alpha=0.3, color='red')
        ax.set_title(f"GSA Block {block_idx + 1} Attention Weights "
                     f"(length {block_length})")
        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(f'attention_map_fold{fold_idx}_{class_names[true_label]}.png',
                dpi=150)
    plt.show()
    print(f"Saved attention visualisation")



if __name__ == "__main__":
    '''
        USAGE
        python3 -m src.dry_run <csv_row_idx>

        where csv_row_idx is an non-negative integer that represents the row
        to use for the dry run. Note that the first (non-header) row has a id
        of 0
    '''
    load_dotenv()
    MODEL_DIR = os.getenv("MODELS_DIR")
    PTBXL = os.getenv("PTBXL_DATASET")
    SPLITS_DIR = "splits"

    FOLD_IDX = 1       # which fold's model to use
    ROW_IDX = int(sys.argv[1])        # which row of the test set to inspect

    # Load test set and pick the row
    test_df = pd.read_csv(os.path.join(SPLITS_DIR, 'test_set.csv'))
    row = test_df.iloc[ROW_IDX]

    filename   = row['filename']
    true_label = int(row['label'])
    ecg_id     = row['ecg_id']

    print(f"Selected test record: ecg_id={ecg_id}, "
          f"filename={filename}, true_label={CLASS_NAMES[true_label]}")

    # Load model
    weights_path = os.path.join(MODEL_DIR, f"best_model_fold{FOLD_IDX + 1}.weights.hdf5")
    model = load_best_model(weights_path)

    # Run prediction and visualisation
    predict_single_record(model, PTBXL, filename, true_label)
    visualise_attention(model, PTBXL, filename, true_label, FOLD_IDX)



