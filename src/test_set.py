from dotenv import load_dotenv
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import numpy as np
import pandas as pd

from src.train import (
    load_best_model,
    preprocess_dataframe,
    load_single_record, 
    preprocess_signal,
    get_attention_mask,
    make_dataset_from_arrays,
    LEAD_II_INDEX, 
    SIGNAL_LENGTH, 
    TARGET_FS,
    CLASS_MAP,
)


CLASS_NAMES = CLASS_MAP.keys()

# ======================== EVALUATION ========================
def evaluate_model_on_test(model, test_signals, test_labels, test_masks, test_has_masks,
                            output_prefix='test'):
    dataset = make_dataset_from_arrays(
        test_signals, test_labels, test_masks, test_has_masks, shuffle=False
    )

    all_labels = []
    all_preds  = []
    all_probs  = []

    for signals, labels, _, _ in dataset:
        pred_probs, _, _ = model(signals, training=False)
        preds = tf.argmax(pred_probs, axis=1).numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(pred_probs.numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    print(f"\n--- Classification Report ({output_prefix}) ---")
    report = classification_report(all_labels, all_preds,
                                     target_names=CLASS_NAMES, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    print(f"--- Confusion Matrix ({output_prefix}) ---")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False)
    plt.title(f"Confusion Matrix — {output_prefix}")
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{output_prefix}.png', dpi=150)
    plt.close()

    return all_labels, all_preds, all_probs, report, cm


def evaluate_single_fold(fold_idx, model_dir, test_signals, test_labels,
                          test_masks, test_has_masks):
    weights_path = os.path.join(model_dir, f"best_model_fold{fold_idx + 1}.weights.hdf5")


    model = load_best_model(weights_path)


    return evaluate_model_on_test(
        model, test_signals, test_labels, test_masks, test_has_masks,
        output_prefix=f'fold{fold_idx + 1}'
    )


def evaluate_ensemble(model_dir, test_signals, test_labels, test_masks, test_has_masks,
                       num_folds=5):
    all_probs = []

    for fold_idx in range(num_folds):
        weights_path = os.path.join(model_dir, f"best_model_fold{fold_idx + 1}.weights.hdf5")

        model = load_best_model(weights_path)

        dataset = make_dataset_from_arrays(
            test_signals, test_labels, test_masks, test_has_masks, shuffle=False
        )

        fold_probs = []
        for signals, _, _, _ in dataset:
            pred_probs, _, _ = model(signals, training=False)
            fold_probs.append(pred_probs.numpy())

        all_probs.append(np.concatenate(fold_probs, axis=0))

    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    print(f"\n--- Classification Report (Ensemble of {num_folds} folds) ---")
    print(classification_report(test_labels, ensemble_preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(test_labels, ensemble_preds)
    print("--- Confusion Matrix (Ensemble) ---")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False)
    plt.title("Confusion Matrix — Ensemble")
    plt.tight_layout()
    plt.savefig('confusion_matrix_ensemble.png', dpi=150)
    plt.close()

    return ensemble_preds, ensemble_probs, cm



if __name__ == "__main__":
    load_dotenv()
    MODEL_DIR = os.getenv("MODELS_DIR")
    PTBXL = os.getenv("PTBXL_DATASET")
    SPLITS_DIR = "splits"

    # Load test set
    test_df = pd.read_csv(os.path.join(SPLITS_DIR, 'test_set.csv'))
    print(f"Test set size: {len(test_df)}")
    for label, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {len(test_df[test_df['label'] == label])}")

    # Preprocess test set (or load cached .npz if you saved one)
    npz_path = os.path.join(SPLITS_DIR, 'test_set_preprocessed.npz')
    if os.path.exists(npz_path):
        print("Loading cached preprocessed test set...")
        data = np.load(npz_path)
        test_signals = data['signals']
        test_labels  = data['labels']
        test_masks   = data['masks']
        test_has_masks = data['has_masks']
    else:
        print("Preprocessing test set...")
        test_signals, test_labels, test_masks, test_has_masks = \
            preprocess_dataframe(test_df, PTBXL)
        np.savez(npz_path, signals=test_signals, labels=test_labels,
                 masks=test_masks, has_masks=test_has_masks)

    # Evaluate each fold individually
    fold_results = []
    for fold_idx in range(5):
        print(f"\n{'='*50}")
        print(f"FOLD {fold_idx + 1}")
        print(f"{'='*50}")
        _, _, _, report, cm = evaluate_single_fold(
            fold_idx, MODEL_DIR, test_signals, test_labels, test_masks, test_has_masks
        )
        fold_results.append(report)

    # Summarise macro F1 across folds
    print(f"\n{'='*50}")
    print("SUMMARY — Macro F1 on Test Set per Fold")
    print(f"{'='*50}")
    for fold_idx, report in enumerate(fold_results):
        macro_f1 = report['macro avg']['f1-score']
        print(f"Fold {fold_idx + 1}: Macro F1 = {macro_f1:.4f}")

    mean_f1 = np.mean([r['macro avg']['f1-score'] for r in fold_results])
    print(f"\nMean Macro F1 across folds: {mean_f1:.4f}")

    # Evaluate ensemble
    print(f"\n{'='*50}")
    print("ENSEMBLE EVALUATION")
    print(f"{'='*50}")
    evaluate_ensemble(MODEL_DIR, test_signals, test_labels, test_masks, test_has_masks)
