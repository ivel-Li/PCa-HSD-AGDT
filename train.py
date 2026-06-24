"""
Training, validation, and evaluation utilities extracted from SWIN-Split.ipynb.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix, recall_score

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

from config import device, batch_size, num_workers, num_epochs, SupCon
from utils.losses import SupConLoss

# ──────────────────────────────────────────────────────────────────────
# Dict State Handling
# ──────────────────────────────────────────────────────────────────────
def get_state_dict_for_saving(model):
    """Remove 'module.' prefix from DataParallel wrapper if present."""
    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()

# ──────────────────────────────────────────────────────────────────────
# Seed
# ──────────────────────────────────────────────────────────────────────
def seed_everything(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """DataLoader worker seed function."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ──────────────────────────────────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────────────────────────────────
def apply_augmentation(adc_batch, dwi_batch, t2_batch, mask_batch=None):
    """
    In-place augmentation applied during training.
    Augmentations (from notebook):
      1) Horizontal flip (50%)
      2) Random rotation (-15°, 15°)
      3) Translation ±10% + scaling 0.9~1.1
    """
    B, S, C, H, W = adc_batch.shape
    adc_batch = adc_batch.clone()
    dwi_batch = dwi_batch.clone()
    t2_batch = t2_batch.clone()
    if mask_batch is not None:
        mask_batch = mask_batch.clone()

    for i in range(B):
        # 1. Horizontal flip (50%)
        if torch.rand(1) < 0.5:
            adc_batch[i] = transforms.functional.hflip(adc_batch[i])
            dwi_batch[i] = transforms.functional.hflip(dwi_batch[i])
            t2_batch[i] = transforms.functional.hflip(t2_batch[i])
            if mask_batch is not None:
                mask_batch[i] = transforms.functional.hflip(mask_batch[i])

        # 2. Random Rotation (-15, 15)
        angle = transforms.RandomRotation.get_params(degrees=(-15, 15))
        adc_batch[i] = transforms.functional.rotate(adc_batch[i], angle)
        dwi_batch[i] = transforms.functional.rotate(dwi_batch[i], angle)
        t2_batch[i] = transforms.functional.rotate(t2_batch[i], angle)
        if mask_batch is not None:
            mask_batch[i] = transforms.functional.rotate(mask_batch[i], angle)

        # 3. Translation + scale
        scale = random.uniform(0.9, 1.1)
        max_dx = 0.1 * W
        max_dy = 0.1 * H
        translate = (
            random.uniform(-max_dx, max_dx),
            random.uniform(-max_dy, max_dy)
        )
        adc_batch[i] = transforms.functional.affine(
            adc_batch[i], angle=0, translate=translate, scale=scale,
            shear=[0, 0], interpolation=transforms.InterpolationMode.BILINEAR
        )
        dwi_batch[i] = transforms.functional.affine(
            dwi_batch[i], angle=0, translate=translate, scale=scale,
            shear=[0, 0], interpolation=transforms.InterpolationMode.BILINEAR
        )
        t2_batch[i] = transforms.functional.affine(
            t2_batch[i], angle=0, translate=translate, scale=scale,
            shear=[0, 0], interpolation=transforms.InterpolationMode.BILINEAR
        )
        if mask_batch is not None:
            mask_batch[i] = transforms.functional.affine(
                mask_batch[i], angle=0, translate=translate, scale=scale,
                shear=[0, 0], interpolation=transforms.InterpolationMode.NEAREST
            )

    return adc_batch, dwi_batch, t2_batch, mask_batch


# ──────────────────────────────────────────────────────────────────────
# Train one epoch
# ──────────────────────────────────────────────────────────────────────
def train(model, train_loader, criterion, optimizer, epoch, supcon_loss=None,
          use_augmentation=True, device=device):
    """
    Train model for one epoch using slices as batch dimension.
    Returns average loss.
    """
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch:3d} Train"):
        adc = batch['adc'].to(device)       # (B, S, 1, H, W)
        dwi = batch['dwi'].to(device)
        t2 = batch['t2'].to(device)
        labels = batch['label'].to(device)  # (B,)
        mask = batch.get('mask')
        if mask is not None:
            mask = mask.to(device)

        # Augmentation
        if use_augmentation:
            adc, dwi, t2, mask = apply_augmentation(adc, dwi, t2, mask)

        optimizer.zero_grad()
        outputs, proj = model(adc, dwi, t2, mask=mask)
        loss = criterion(outputs, labels)

        # Supervised contrastive loss
        if supcon_loss is not None:
            alpha = 0.5
            labels_rep = labels.unsqueeze(1).repeat(1, 16).view(-1)  # (B*16,)
            loss_supcon = supcon_loss(proj, labels_rep)
            loss = loss + alpha * loss_supcon

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# ──────────────────────────────────────────────────────────────────────
# Validate
# ──────────────────────────────────────────────────────────────────────
def validate(model, val_loader, criterion, supcon_loss=None, device=device):
    """
    Validate model on validation set.
    Returns (avg_loss, accuracy_percent).
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            adc = batch['adc'].to(device)
            dwi = batch['dwi'].to(device)
            t2 = batch['t2'].to(device)
            labels = batch['label'].to(device)
            mask = batch.get('mask')
            if mask is not None:
                mask = mask.to(device)

            outputs, proj = model(adc, dwi, t2, mask=mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return total_loss / len(val_loader), accuracy


# ──────────────────────────────────────────────────────────────────────
# Joint Recall for cancer classes (2, 3)
# ──────────────────────────────────────────────────────────────────────
def joint_recall(y_true, y_pred, positive_labels=(2, 3)):
    """
    Compute recall for a set of positive labels.
    A prediction is correct if it falls into positive_labels,
    regardless of which specific label inside that set.
    """
    y_true_bin = [1 if y in positive_labels else 0 for y in y_true]
    y_pred_bin = [1 if y in positive_labels else 0 for y in y_pred]
    return recall_score(y_true_bin, y_pred_bin)


# ──────────────────────────────────────────────────────────────────────
# Evaluate (full report + confusion matrix)
# ──────────────────────────────────────────────────────────────────────
def evaluate(model, data_loader, dataset_name, save_path=None,
             plt_title='None', plt_show=False, num_classes=4, device=device):
    """
    Full evaluation: accuracy, per-class recall, confusion matrix, cancer joint recall.
    """
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating on {dataset_name}"):
            adc = batch['adc'].to(device)
            dwi = batch['dwi'].to(device)
            t2 = batch['t2'].to(device)
            labels = batch['label'].to(device)
            mask = batch.get('mask')
            if mask is not None:
                mask = mask.to(device)

            outputs, proj = model(adc, dwi, t2, mask=mask)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    print(f"\n--- Evaluation Report for {dataset_name} ---")
    target_names = [f'Class {i}' for i in range(num_classes)]
    report_dict = classification_report(
        all_labels, all_predictions, target_names=target_names, output_dict=True
    )

    acc = report_dict["accuracy"]
    class_acc = {
        str(c): report_dict[f'Class {c}']['recall'] for c in range(num_classes)
    }
    cancer_recall = joint_recall(all_labels, all_predictions)
    print(classification_report(all_labels, all_predictions, target_names=target_names))

    # Confusion matrix plot
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {plt_title}')
    plt.colorbar()

    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if save_path is not None:
        plt.savefig(f"{save_path}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}_confusion_matrix.png")

    if plt_show:
        plt.show()
    plt.close()

    return acc, class_acc, cancer_recall


