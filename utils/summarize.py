#!/usr/bin/env python3
"""
Re-evaluate best models across folds for a given experiment.
Computes patient-level accuracy, per-class recall, and cancer joint recall.

Usage:
    conda activate picai
    python utils/summarize.py run/AGDT/postmask/seed42
    python utils/summarize.py run/AGDT/postmask/seed42 --device cuda:0

For multiple seeds:
    for d in run/AGDT/postmask/seed*/; do
        python utils/summarize.py "$d"
    done
"""

import os
import sys
import json
import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import recall_score

from data.dataset import MRIDataset
from data.split import build_splits
from models.model_factory import build_model
from train import get_state_dict_for_saving


# ──────────────────────────────────────────────────────────────────────
# Joint recall (cancer = label 2 or 3)
# ──────────────────────────────────────────────────────────────────────
def joint_recall(y_true, y_pred, positive_labels=(2, 3)):
    y_true_bin = [1 if y in positive_labels else 0 for y in y_true]
    y_pred_bin = [1 if y in positive_labels else 0 for y in y_pred]
    return recall_score(y_true_bin, y_pred_bin)


# ──────────────────────────────────────────────────────────────────────
# Evaluate a model on a test DataLoader
# ──────────────────────────────────────────────────────────────────────
def evaluate_model(model, data_loader, device, num_classes=4):
    """
    Run inference and collect per-patient predictions.
    Returns (all_labels, all_predictions) numpy arrays.
    """
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            adc = batch['adc'].to(device)
            dwi = batch['dwi'].to(device)
            t2 = batch['t2'].to(device)
            labels = batch['label'].to(device)
            mask = batch.get('mask')
            if mask is not None:
                mask = mask.to(device)

            outputs, _ = model(adc, dwi, t2, mask=mask)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    return np.array(all_labels), np.array(all_predictions)


# ──────────────────────────────────────────────────────────────────────
# Read best accuracy from training log (for reference)
# ──────────────────────────────────────────────────────────────────────
def best_acc_from_log(log_path):
    """Return the maximum test accuracy (%) from a training log JSON."""
    with open(log_path) as f:
        data = json.load(f)
    accs = data["test_acc"]
    best = max(accs)
    best_epoch = data["epoch"][accs.index(best)]
    return best, best_epoch


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def summarize(exp_path, device="cuda:0"):
    exp_path = os.path.abspath(exp_path) if not os.path.isabs(exp_path) else exp_path

    # Parse seed and model info from path
    parts = exp_path.rstrip("/").split("/")
    run_name = parts[-1]  # e.g. seed42, seed202, 1
    trick_num = int(parts[-2]) if parts[-2].isdigit() else 5
    model_name = parts[-3]

    # Determine seed from run_name
    seed_map = {
        "seed42": 42, "seed202": 202, "seed2003": 2003, "seed527": 527,
        "1": 42,
    }
    seed = seed_map.get(run_name, None)
    if seed is None:
        print(f"[ERROR] Unknown run name '{run_name}', cannot determine seed")
        sys.exit(1)

    print(f"Experiment: {exp_path}")
    print(f"  Model: {model_name}, trick_num: {trick_num}, run: {run_name}, seed: {seed}")

    # ── 1. Find fold directories ────────────────────────────────────
    fold_dirs = sorted([
        d for d in os.listdir(exp_path)
        if d.startswith("fold_") and os.path.isdir(os.path.join(exp_path, d))
    ])
    if not fold_dirs:
        print(f"[ERROR] No fold directories found in {exp_path}")
        sys.exit(1)

    # ── 2. Load dataset info ────────────────────────────────────────
    HDF5_PATH = "./dataset/patients_dataset_v1.0.h5"
    with h5py.File(HDF5_PATH, "r") as f:
        n = len(f)
        patient_indices = np.arange(n)
        labels_all = np.array([f[str(i)]["label"][()] for i in patient_indices])

    print(f"Total patients: {n}")
    print(f"Label distribution: {np.bincount(labels_all)}")

    selected_indices = patient_indices
    labels_selected = labels_all

    # ── 3. Per-fold evaluation ─────────────────────────────────────
    fold_accs = []
    fold_class_accs = []
    fold_cancer_recalls = []

    for fold, _, test_idx in build_splits(
        selected_indices, labels_selected, num_splits=5, seed=seed
    ):
        fold_path = os.path.join(exp_path, f"fold_{fold}")
        print(f"\n{'='*60}")
        print(f"Fold {fold}")
        print(f"{'='*60}")

        # Load best model checkpoint
        model = build_model(model_name)
        best_path = os.path.join(fold_path, f"{run_name}_best_model.pth")
        if not os.path.exists(best_path):
            print(f"[WARN] Best model not found: {best_path}")
            continue
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state)
        model = model.to(device)

        # Create test dataset with the CORRECT split
        use_rescaled = True
        test_dataset = MRIDataset(HDF5_PATH, test_idx, use_rescaled=use_rescaled)
        test_loader = DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
        )

        # Evaluate
        all_labels, all_preds = evaluate_model(model, test_loader, device)

        # Compute metrics
        acc = np.mean(all_labels == all_preds)
        class_recalls = {}
        for c in range(4):
            mask = all_labels == c
            if mask.sum() > 0:
                class_recalls[str(c)] = (all_preds[mask] == c).sum() / mask.sum()
            else:
                class_recalls[str(c)] = 0.0

        cancer_rec = joint_recall(all_labels, all_preds)

        fold_accs.append(acc)
        fold_class_accs.append(class_recalls)
        fold_cancer_recalls.append(cancer_rec)

        print(f"  Test accuracy:     {acc*100:.2f}%")
        print(f"  Per-class recall:  {[f'{class_recalls[str(c)]*100:.1f}%' for c in range(4)]}")
        print(f"  Cancer joint recall: {cancer_rec:.4f}")

        # Log accuracy from training log for comparison
        log_files = [f for f in os.listdir(fold_path) if f.endswith("_training_log.json")]
        if log_files:
            log_path = os.path.join(fold_path, log_files[0])
            best_log_acc, best_epoch = best_acc_from_log(log_path)
            print(f"  Training log best acc: {best_log_acc:.2f}% @ epoch {best_epoch}")
            print(f"  Log acc vs recomputed: {best_log_acc:.2f}% vs {acc*100:.2f}%")

    # ── 4. Aggregate results ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"AGGREGATED RESULTS: {run_name} (seed={seed})")
    print(f"{'='*60}")

    if len(fold_accs) == 0:
        print("[ERROR] No folds evaluated successfully")
        return

    acc_arr = np.array(fold_accs) * 100
    cr_arr = np.array(fold_cancer_recalls)

    print(f"\n  Per-fold accuracy:       {[f'{a:.2f}%' for a in acc_arr]}")
    print(f"  Accuracy Mean ± Std:     {acc_arr.mean():.2f}% ± {acc_arr.std():.2f}%")
    print(f"\n  Per-fold cancer recall:  {[f'{r:.4f}' for r in fold_cancer_recalls]}")
    print(f"  Cancer recall Mean ± Std: {cr_arr.mean():.4f} ± {cr_arr.std():.4f}")

    avg_per_class = {}
    for c in range(4):
        vals = [fc[str(c)] * 100 for fc in fold_class_accs]
        avg_per_class[f"Class {c}"] = f"{np.mean(vals):.1f}%"
    print(f"\n  Average per-class recall: {avg_per_class}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-evaluate best models")
    parser.add_argument("exp_path", type=str,
                        help="Path to experiment folder, e.g. run/AGDT/postmask/seed42")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device (default: cuda:0)")
    args = parser.parse_args()

    summarize(args.exp_path, device=args.device)
