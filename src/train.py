
import numpy as np
import tensorflow as tf
import pandas as pd
import wfdb
import ast
import os

from scipy.signal import butter, filtfilt, resample
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from src.gam import Attention_Maps, GSA
from src.data_reader.data_reader import PTB_XL_Reader
from src.data_organisation.ptbxl_organisation import ptbxl_cond_to_ids

tf.get_logger().setLevel('ERROR')  # only shows ERROR and above


# ======================== CONSTANTS ========================
TARGET_FS = 120
SIGNAL_LENGTH = 1200        # 10s * 120Hz
LEAD_II_INDEX = 1           # 0-indexed in PTB-XL
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 50
ALPHA = 0.6                 # weighting between classification and attention loss

CLASS_MAP = {'NORM': 0, 'LBBB': 1, 'RBBB': 2, '1dAVB': 3}
NORM_SUBSAMPLE_TARGET = 1600

# Class weights to handle residual imbalance after subsampling
# NORM ~1600, others ~535 each -> roughly 3:1
# These are approximate; tune after seeing your exact counts
CLASS_WEIGHTS = {
    0: 1.0,   # NORM
    1: 3.0,   # LBBB
    2: 3.0,   # RBBB
    3: 3.0,   # 1dAVB
}

# ======================== PREPROCESSING ========================
def butter_bandpass_filter(signal, lowcut=0.67, highcut=45.0, fs=500, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)

def downsample_signal(signal, original_fs, target_fs):
    num_target_samples = int(signal.shape[0] * target_fs / original_fs)
    return resample(signal, num_target_samples, axis=0)

def zscore_normalise(signal):
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0)
    std[std == 0] = 1.0     # avoid division by zero for flat leads
    return (signal - mean) / std

def preprocess_signal(raw_signal, original_fs):
    '''
    Full preprocessing pipeline:
    1. Butterworth bandpass filter
    2. Downsample to TARGET_FS
    3. Z-score normalise
    4. Pad or truncate to SIGNAL_LENGTH
    Returns: (SIGNAL_LENGTH, 12) float32 array
    '''
    filtered = butter_bandpass_filter(raw_signal, fs=original_fs)
    downsampled = downsample_signal(filtered, original_fs, TARGET_FS)
    normalised = zscore_normalise(downsampled)

    # Pad or truncate to fixed length
    result = np.zeros((SIGNAL_LENGTH, 12), dtype=np.float32)
    actual_length = min(normalised.shape[0], SIGNAL_LENGTH)
    result[:actual_length, :] = normalised[:actual_length, :]

    return result

def preprocess_dataframe(df, database_path):
    signals = []
    att_masks = []
    has_masks = []
    valid_labels = []

    for _, row in df.iterrows():
        try:
            raw_signal, fs = load_single_record(database_path, row['filename'])
            processed = preprocess_signal(raw_signal, fs)
            lead_ii = processed[:, LEAD_II_INDEX]
            mask = get_attention_mask(lead_ii, row['label'])

            if mask is None:
                att_mask = np.zeros((SIGNAL_LENGTH, 1), dtype=np.float32)
                has_mask = np.float32(0.0)
            else:
                att_mask = mask.astype(np.float32)
                has_mask = np.float32(1.0)

            signals.append(processed)
            att_masks.append(att_mask)
            has_masks.append(has_mask)
            valid_labels.append(row['label'])

        except Exception as e:
            print(f"Skipping record {row['ecg_id']}: {e}")
            continue

    return (
        np.array(signals,    dtype=np.float32),
        np.array(valid_labels, dtype=np.int32),
        np.array(att_masks,  dtype=np.float32),
        np.array(has_masks,  dtype=np.float32),
    )


# ======================== DATASET PREPARATION ========================
def load_and_filter_records(reader, class_ids, test_size=0.15):
    '''
    Loads full CSV, filters to class IDs, subsamples NORM,
    returns a dataframe with ecg_id, label, filename columns
    '''
    csv = reader.get_csv()

    norm_ids = set(class_ids['NORM'])
    lbbb_ids = set(class_ids['LBBB'])
    rbbb_ids = set(class_ids['RBBB'])
    avb_ids  = set(class_ids['1dAVB'])

    records = []
    for ecg_id, row in csv.iterrows():
        if ecg_id in norm_ids:
            records.append({'ecg_id': ecg_id, 'label': 0,
                            'filename': row['filename_hr']})
        elif ecg_id in lbbb_ids:
            records.append({'ecg_id': ecg_id, 'label': 1,
                            'filename': row['filename_hr']})
        elif ecg_id in rbbb_ids:
            records.append({'ecg_id': ecg_id, 'label': 2,
                            'filename': row['filename_hr']})
        elif ecg_id in avb_ids:
            records.append({'ecg_id': ecg_id, 'label': 3,
                            'filename': row['filename_hr']})

    df = pd.DataFrame(records)

    # Subsample NORM
    norm_df = df[df['label'] == 0].sample(
        n=NORM_SUBSAMPLE_TARGET, random_state=42
    )
    other_df = df[df['label'] != 0]
    df = pd.concat([norm_df, other_df]).reset_index(drop=True)

    # Train-Test split
    dev_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=42
    )
    dev_df  = dev_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)


    print("Dev set class counts:")
    for label, name in enumerate(['NORM', 'LBBB', 'RBBB', '1dAVB']):
        print(f"  {name}: {len(dev_df[dev_df['label'] == label])}")

    print("Test set class counts:")
    for label, name in enumerate(['NORM', 'LBBB', 'RBBB', '1dAVB']):
        print(f"  {name}: {len(test_df[test_df['label'] == label])}")

    return dev_df, test_df

def load_single_record(database_path, filename):
    record = wfdb.rdrecord(database_path + filename)
    return record.p_signal, record.fs

# ======================== ATTENTION MASK GENERATION ========================
attention_maps = Attention_Maps()

def get_attention_mask(preprocessed_lead_ii, label):
    '''
    Generate attention mask from preprocessed lead II signal.
    NORM (label=0) returns None — no attention supervision.
    LBBB/RBBB (labels 1,2) use QRS mask.
    1dAVB (label=3) uses PR mask.
    '''
    if label == 0:
        return None
    elif label == 1 or label == 2:
        return attention_maps.generate_qrs_mask(preprocessed_lead_ii, TARGET_FS)
    elif label == 3:
        return attention_maps.generate_pr_mask(preprocessed_lead_ii, TARGET_FS)

# ======================== TF DATASET ========================
def make_dataset_from_arrays(signals, labels, att_masks, has_masks, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(
        (signals, labels, att_masks, has_masks)
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(signals))
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset


# ======================== TRAINING STEP ========================
def train_step(model, optimizer, signals, labels, attention_masks, has_mask_flags):
    with tf.GradientTape() as tape:
        pred_probs, att_weights, att_logits = model(signals, training=True)

        # One-hot encode labels for classification loss
        labels_onehot = tf.one_hot(labels, NUM_CLASSES)

        # Apply class weights
        sample_weights = tf.reduce_sum(
            tf.constant([[CLASS_WEIGHTS[i] for i in range(NUM_CLASSES)]],
                        dtype=tf.float32) * labels_onehot,
            axis=1
        )

        # Classification loss (weighted)
        loss_cls = model.classification_loss(labels_onehot, pred_probs)
        loss_cls = tf.reduce_mean(loss_cls * sample_weights)

        # Attention loss — only for samples with a mask
        # has_mask_flags is (B,); use it to zero out NORM contributions
        loss_att_raw = model.attention_loss(attention_masks, att_logits)

        # Scale attention loss by mask flags so NORM doesn't contribute
        loss_att = loss_att_raw * tf.reduce_mean(has_mask_flags)

        total_loss = ALPHA * loss_cls + (1 - ALPHA) * loss_att

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss, loss_cls, loss_att

def val_step(model, signals, labels, attention_masks, has_mask_flags):
    pred_probs, att_weights, att_logits = model(signals, training=False)
    labels_onehot = tf.one_hot(labels, NUM_CLASSES)
    loss_cls = model.classification_loss(labels_onehot, pred_probs)
    loss_att = model.attention_loss(attention_masks, att_logits)
    loss_att = loss_att * tf.reduce_mean(has_mask_flags)
    total_loss = ALPHA * loss_cls + (1 - ALPHA) * loss_att
    return total_loss, pred_probs

# ======================== METRICS ========================
def compute_f1_per_class(all_labels, all_preds, num_classes=NUM_CLASSES):
    f1_scores = []
    for c in range(num_classes):
        tp = np.sum((all_preds == c) & (all_labels == c))
        fp = np.sum((all_preds == c) & (all_labels != c))
        fn = np.sum((all_preds != c) & (all_labels == c))
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)
    return f1_scores

# ========= SAVING TRAIN-VALIDATION-TEST INFO ============
def save_split_info(test_df, fold_splits, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    test_df.to_csv(os.path.join(output_dir, 'test_set.csv'), index=False)
    
    for fold_idx, (train_df, val_df) in enumerate(fold_splits):
        train_df.to_csv(os.path.join(output_dir, f'fold{fold_idx+1}_train.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, f'fold{fold_idx+1}_val.csv'), index=False)

# ======================== TRAINING LOOP ========================
def run_cross_validation(reader, database_path, class_ids):
    df, test_df = load_and_filter_records(reader, class_ids)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels_array = df['label'].values

    fold_results = []
    fold_splits = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(df)), labels_array)
    ):
        print(f"\n{'='*50}")
        print(f"FOLD {fold_idx + 1} / 5")
        print(f"{'='*50}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)
        
        fold_splits.append((train_df, val_df))

        print(f"Train size: {len(train_df)} | Val size: {len(val_df)}")

        print("Preprocessing train set...")
        train_signals, train_labels, train_masks, train_has_masks = \
            preprocess_dataframe(train_df, database_path)

        print("Preprocessing val set...")
        val_signals, val_labels, val_masks, val_has_masks = \
            preprocess_dataframe(val_df, database_path)

        train_dataset = make_dataset_from_arrays(
            train_signals, train_labels, train_masks, train_has_masks, shuffle=True
        )
        val_dataset = make_dataset_from_arrays(
            val_signals, val_labels, val_masks, val_has_masks, shuffle=False
        )

        model = GSA()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        best_val_loss = float('inf')
        best_val_f1   = 0.0

        for epoch in range(EPOCHS):
            # --- Training ---
            train_losses = []
            for signals, labels, att_masks, has_masks in train_dataset:
                t_loss, t_cls, t_att = train_step(
                    model, optimizer, signals, labels, att_masks, has_masks
                )
                train_losses.append(t_loss.numpy())

            # --- Validation ---
            val_losses  = []
            all_labels  = []
            all_preds   = []

            for signals, labels, att_masks, has_masks in val_dataset:
                v_loss, pred_probs = val_step(
                    model, signals, labels, att_masks, has_masks
                )
                val_losses.append(v_loss.numpy())
                preds = tf.argmax(pred_probs, axis=1).numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

            all_labels = np.array(all_labels)
            all_preds  = np.array(all_preds)
            f1_scores  = compute_f1_per_class(all_labels, all_preds)
            macro_f1   = np.mean(f1_scores)

            mean_train_loss = np.mean(train_losses)
            mean_val_loss   = np.mean(val_losses)

            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train Loss: {mean_train_loss:.4f} | "
                  f"Val Loss: {mean_val_loss:.4f} | "
                  f"Macro F1: {macro_f1:.4f} | "
                  f"F1 [NORM:{f1_scores[0]:.3f} "
                  f"LBBB:{f1_scores[1]:.3f} "
                  f"RBBB:{f1_scores[2]:.3f} "
                  f"1dAVB:{f1_scores[3]:.3f}]")

            if macro_f1 > best_val_f1:
                best_val_f1   = macro_f1
                best_val_loss = mean_val_loss
                model.save_weights(f'best_model_fold{fold_idx+1}.weights.h5')
                # model.save(f'best_model_fold{fold_idx+1}.keras')             # for retraining
                print(f"  -> Saved best model (Macro F1: {best_val_f1:.4f})")

        fold_results.append({
            'fold': fold_idx + 1,
            'best_val_f1': best_val_f1,
            'best_val_loss': best_val_loss,
            'path': f'best_model_fold{fold_idx+1}.weights.h5'
        })
    
    save_split_info(test_df, fold_splits, output_dir='splits')


    print(f"\n{'='*50}")
    print("CROSS VALIDATION COMPLETE")
    print(f"{'='*50}")
    for r in fold_results:
        print(f"Fold {r['fold']}: Best Val F1 = {r['best_val_f1']:.4f}")
    mean_f1 = np.mean([r['best_val_f1'] for r in fold_results])
    print(f"Mean Macro F1 across folds: {mean_f1:.4f}")

    return test_df, fold_results



# ================================= MODEL LOAD =================================

def load_best_model(weights_path):
    # 1. Clear session to avoid overlapping layer increment names
    tf.keras.backend.clear_session()

    # 2. Instantiate your custom model architecture
    model = GSA()

    dummy_input = tf.zeros((1, SIGNAL_LENGTH, 12), dtype=tf.float32)
    _ = model(dummy_input, training=False)
    
    # 4. Load weights by topology order instead of literal string names
    try:
        model.load_weights(weights_path, by_name=False)
        print(f"Successfully loaded weights topologically from: {weights_path}")
    except Exception as e:
        print("Standard topological load failed. Attempting fallback mapping...")
        # Fallback option: If structural tracking is nested inside an outer layer container
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        
    return model 

# ================================= TESTING SET ================================
def evaluate_fold(model, signals, labels, masks, has_masks, confusion_matrix_file_name="confusion_matrix.png"):
    dataset = make_dataset_from_arrays(signals, labels, masks, has_masks, shuffle=False)

    all_labels = []
    all_preds  = []
    all_probs  = []

    for batch_signals, batch_labels, _, _ in dataset:
        pred_probs, _, _ = model(batch_signals, training=False)
        preds = tf.argmax(pred_probs, axis=1).numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_labels.numpy())
        all_probs.extend(pred_probs.numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    class_names = ['NORM', 'LBBB', 'RBBB', '1dAVB']

    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(confusion_matrix_file_name, dpi=150)

    return all_labels, all_preds, all_probs


def evaluate_on_test_set(test_df, database_path, fold_model_paths, best_fold_idx):
    test_signals, test_labels, test_masks, test_has_masks = \
        preprocess_dataframe(test_df, database_path)

    test_dataset = make_dataset_from_arrays(
        test_signals, test_labels, test_masks, test_has_masks, shuffle=False
    )

    # Option A: evaluate the single best fold model
    best_model = load_best_model(fold_model_paths[best_fold_idx])
    evaluate_fold(best_model, test_signals, test_labels, test_masks, test_has_masks)

    # Option B: ensemble all 5 fold models by averaging predictions
    all_probs = []
    for path in fold_model_paths:
        model = load_best_model(path)
        probs_list = []
        for signals, labels, _, _ in test_dataset:
            pred_probs, _, _ = model(signals, training=False)
            probs_list.append(pred_probs.numpy())
        all_probs.append(np.concatenate(probs_list, axis=0))

    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    class_names = ['NORM', 'LBBB', 'RBBB', '1dAVB']
    print(classification_report(test_labels, ensemble_preds, target_names=class_names))

# ======================== ENTRY POINT ========================
if __name__ == "__main__":
    load_dotenv()
    DATABASE_PATH = os.getenv("PTBXL_DATASET")

    # Your cleaned class ID lists from earlier
    class_ids = ptbxl_cond_to_ids()

    reader = PTB_XL_Reader(DATABASE_PATH)
    test_df, fold_results = run_cross_validation(reader, DATABASE_PATH, class_ids)
    
    # Has keys             
        # 'fold'
        # 'best_val_f1'
        # 'best_val_loss'
        # 'path'
    best_fold = max(fold_results, key=lambda r: r['best_val_f1'])
    fold_paths = [r['path'] for r in fold_results]
        

    evaluate_on_test_set(test_df, DATABASE_PATH, fold_paths, best_fold['fold'] - 1)
