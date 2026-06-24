#!/usr/bin/env python3
"""
Main entry point for PCa-HSD-LSDT training + evaluation.

Mirrors the full pipeline from SWIN-Split.ipynb:
  1. Load HDF5 dataset, read labels
  2. K-fold stratified split
  3. Per-fold: train → validate → save best model
  4. Post-training evaluation across folds (confusion matrices + plots)
  5. Plot training curves (loss + accuracy)

Usage:
    python main.py                                    # use default config.py
    python main.py --config config                    # same as above
    python main.py --config config_resnet             # use config_resnet.py
    python main.py --config /path/to/custom_config.py # absolute/relative path
    python main.py --gpu cuda:1                       # override GPU device
    python main.py --gpu cpu                          # run on CPU
"""

import os
import sys
import json
import argparse
import importlib.util
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════
# Config Loading (supports --config and --gpu)
# ═══════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="PCa-HSD-LSDT training")
parser.add_argument("--config", type=str, default="config",
                    help="Config module name or path (default: config)")
parser.add_argument("--gpu", type=str, default=None,
                    help="Override GPU device (e.g. cuda:0, cuda:1, cpu)")
parser.add_argument("--seed", type=int, default=None,
                    help="Override random seed (default: config.seed)")
parser.add_argument("--run", type=str, default=None,
                    help="Override run id (default: config.run). "
                         "Use e.g. --run seed42 to avoid overwriting other seeds.")
args, _ = parser.parse_known_args()

# Resolve config module path
cfg_name = args.config
if cfg_name.endswith(".py"):
    cfg_name = cfg_name[:-3]
cfg_name = cfg_name.replace("/", ".")

config = importlib.import_module(cfg_name)

# Override device if --gpu is provided
if args.gpu is not None:
    config.device = args.gpu
    if args.gpu == "cpu":
        config.DataParallel = False

if args.seed is not None:
    config.seed = args.seed

if args.run is not None:
    config.run = args.run

# Re-export all config values for backward compatibility
model_name = config.model_name
run = config.run
trick_num = config.trick_num
DataParallel = config.DataParallel
device = config.device
num_epochs = config.num_epochs
batch_size = config.batch_size
num_splits = config.num_splits
DATA_EXTAND = config.DATA_EXTAND
SupCon = config.SupCon
seed = config.seed
num_workers = config.num_workers
HDF5_PATH = config.HDF5_PATH

# Recompute save_path after overrides (run, seed, etc.)
save_path = f"./run/{model_name}/{trick_num}/{run}"
setup_directories = config.setup_directories

# ── Project modules ──────────────────────────────────────────────────
from data.dataset import MRIDataset
from data.split import build_splits
from models.model_factory import build_model
from train import (
    seed_everything,
    seed_worker,
    train,
    validate,
    evaluate,
    get_state_dict_for_saving,
)
from utils.losses import SupConLoss


# ═══════════════════════════════════════════════════════════════════════
# 1.  Setup
# ═══════════════════════════════════════════════════════════════════════
seed_everything(seed)
setup_directories()

g = torch.Generator()
g.manual_seed(seed)

print(f"Config: {cfg_name}")
print(f"Model: {model_name}")
print(f"Save path: {save_path}")
print(f"Device: {device}")
print(f"DataParallel: {DataParallel}")
print(f"SupCon: {SupCon}")

# ═══════════════════════════════════════════════════════════════════════
# 2.  Load HDF5 dataset
# ═══════════════════════════════════════════════════════════════════════
print(f"\nLoading dataset from: {HDF5_PATH}")

with h5py.File(HDF5_PATH, "r") as f:
    n = len(f)
    print(f"Total patients: {n}")
    patient_indices = np.arange(n)
    # Labels for ALL patients
    labels_all = np.array([f[str(i)]["label"][()] for i in patient_indices])

print("Label distribution:", np.bincount(labels_all))

# For the default pipeline we use all patients (no early-stage filtering).
# The notebook filtered with `true_indices`; we replicate that with a simple
# `selected_indices = patient_indices` approach. If you need the early-stage
# filter, pass `true_indices` from the notebook logic instead.
selected_indices = patient_indices
labels_selected = labels_all

# ═══════════════════════════════════════════════════════════════════════
# 3.  K-Fold Cross-Validation / Training
# ═══════════════════════════════════════════════════════════════════════
best_overall_acc = 0.0

for fold, train_idx, test_idx in build_splits(
    selected_indices, labels_selected, num_splits=num_splits, seed=seed
):
    print(f"\n{'='*60}")
    print(f"Fold {fold}")
    print(f"{'='*60}")

    if DATA_EXTAND:
        # In the notebook, false_indices were patients NOT in true_indices
        # (i.e., the complement set). Since we use all patients here,
        # DATA_EXTAND is effectively a no-op unless you've filtered.
        pass

    train_labels = labels_all[train_idx]
    test_labels = labels_all[test_idx]
    print("Train class counts:", np.bincount(train_labels, minlength=4))
    print("Test class counts: ", np.bincount(test_labels, minlength=4))

    # ── Datasets & DataLoaders ────────────────────────────────────
    use_rescaled = True
    train_dataset = MRIDataset(HDF5_PATH, train_idx, use_rescaled=use_rescaled)
    test_dataset = MRIDataset(HDF5_PATH, test_idx, use_rescaled=use_rescaled)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # ── Model ─────────────────────────────────────────────────────
    model = build_model(model_name)
    model = model.to(device)

    if DataParallel and torch.cuda.device_count() > 1:
        print(f"[INFO] Using DataParallel with {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model, device_ids=[0, 1])

    # If MRIClassifier, freeze mask_generator
    if model_name == "MRIClassifier":
        if DataParallel:
            for param in model.module.mask_generator.parameters():
                param.requires_grad = False
            model.module.mask_generator.eval()
        else:
            for param in model.mask_generator.parameters():
                param.requires_grad = False
            model.mask_generator.eval()

    # ── Optimizer & Loss ──────────────────────────────────────────
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
    )
    criterion = nn.CrossEntropyLoss()
    supcon_loss = SupConLoss(temperature=0.07) if SupCon else None

    # ── Training log ──────────────────────────────────────────────
    logs = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "test_acc": [],
    }

    # ── Training loop ─────────────────────────────────────────────
    best_test_acc = 0.0
    for epoch in range(num_epochs):
        tr_loss = train(model, train_loader, criterion, optimizer, epoch,
                        supcon_loss=supcon_loss, device=device)
        te_loss, te_acc = validate(model, test_loader, criterion,
                                   supcon_loss=supcon_loss, device=device)

        logs["epoch"].append(epoch)
        logs["train_loss"].append(tr_loss)
        logs["test_loss"].append(te_loss)
        logs["test_acc"].append(te_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d}/{num_epochs}  "
            f"Train Loss: {tr_loss:.4f}  "
            f"Test Loss:  {te_loss:.4f}  "
            f"Test Acc:   {te_acc:.2f}%  "
            f"LR: {current_lr:.2e}"
        )

        # Save checkpoint every 100 epochs when LR <= 1e-4
        if current_lr <= 1e-4 and (epoch + 1) % 100 == 0:
            torch.save(
                get_state_dict_for_saving(model),
                f"{save_path}/fold_{fold}/{run}_{epoch}.pth",
            )
            print(f"  → Saved snapshot at epoch {epoch+1}")

        # Save best model
        if te_acc >= best_test_acc:
            best_test_acc = te_acc
            torch.save(
                get_state_dict_for_saving(model),
                f"{save_path}/fold_{fold}/{run}_best_model.pth",
            )
            print(f"  → Saved best model (acc={te_acc:.2f}%)")

    # Save training log
    log_path = f"{save_path}/fold_{fold}/{run}_training_log.json"
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=4)
    print(f"Training log saved to {log_path}")

    if best_test_acc > best_overall_acc:
        best_overall_acc = best_test_acc


# ═══════════════════════════════════════════════════════════════════════
# 4.  Post-Training Evaluation
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Post-Training Evaluation")
print(f"{'='*60}")

model = build_model(model_name)
acc_list = []
all_class_acc = []
all_cancer_recall = []

for fold, train_idx, test_idx in build_splits(
    selected_indices, labels_selected, num_splits=num_splits, seed=seed
):
    test_labels = labels_all[test_idx]

    use_rescaled = True
    test_dataset = MRIDataset(HDF5_PATH, test_idx, use_rescaled=use_rescaled)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Load best model weights
    best_path = f"{save_path}/fold_{fold}/{run}_best_model.pth"
    if not os.path.exists(best_path):
        print(f"[WARN] Best model not found: {best_path}")
        continue

    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)

    if DataParallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])

    plt_show = fold == 0
    acc, class_acc, cancer_recall = evaluate(
        model, test_loader, "Test Set",
        save_path=f"{save_path}/fold_{fold}/fold_{fold}",
        plt_title=f"fold_{fold}_test_confusion_matrix",
        plt_show=plt_show,
        device=device,
    )
    acc_list.append(acc)
    all_class_acc.append(class_acc)
    all_cancer_recall.append(cancer_recall)

# ── Aggregate results ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Aggregated Results")
print(f"{'='*60}")
print("Test Accuracies :", [f"{a*100:.2f}%" for a in acc_list])
avg_acc_val = np.mean(acc_list) * 100
std_acc_val = np.std(acc_list) * 100
print(f"Average Accuracy: {avg_acc_val:.2f}% ± {std_acc_val:.2f}%")

print("\nCancer Joint Recalls:", all_cancer_recall)
avg_cr = np.mean(all_cancer_recall)
std_cr = np.std(all_cancer_recall)
print(f"Average Cancer Recall: {avg_cr:.4f} ± {std_cr:.4f}")

avg_acc_per_class = {}
for c in range(4):
    vals = [fold_acc[str(c)] for fold_acc in all_class_acc]
    avg_acc_per_class[str(c)] = np.mean(vals)
print("\nAverage per-class recall across folds:", avg_acc_per_class)


# ═══════════════════════════════════════════════════════════════════════
# 5.  Training Curves
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Plotting Training Curves")
print(f"{'='*60}")

for split_index in range(num_splits):
    log_path = f"{save_path}/fold_{split_index}/{run}_training_log.json"
    if not os.path.exists(log_path):
        print(f"[WARN] Log not found: {log_path}")
        continue

    with open(log_path, "r") as f:
        history = json.load(f)

    epochs = history["epoch"]
    train_loss = history["train_loss"]
    val_loss = history["test_loss"]
    val_acc = history["test_acc"]

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss", marker="o")
    plt.plot(epochs, val_loss, label="Test Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1.5)
    plt.title(f"{model_name} - Training and Validation Loss (Fold {split_index})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}/fold_{split_index}/Tra_Val_Loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_acc, label="Test Accuracy", marker="o", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model_name} - Validation Accuracy (Fold {split_index})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}/fold_{split_index}/Tra_Val_Acc.png", dpi=300, bbox_inches="tight")
    plt.close()

print(f"\nDone! Results saved to {save_path}")