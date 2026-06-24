"""
Dataset splitting utilities – mirrors build_splits() from SWIN-Split.ipynb.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def build_splits(patient_indices, labels, num_splits, test_size=0.2, seed=42):
    """
    Build train/test splits (k-fold or single split).

    Parameters
    ----------
    patient_indices : array-like
        Patient identifiers (e.g., integer indices or strings).
    labels : array-like
        Corresponding class labels (used for stratified split).
    num_splits : int
        Number of folds. If >= 2, uses StratifiedKFold.
        If == 1, uses a single train_test_split.
    test_size : float
        Fraction to hold out as test when num_splits == 1.
    seed : int
        Random seed for reproducibility.

    Yields
    ------
    (fold, train_idx, test_idx)
        fold      : int – fold number (0-indexed)
        train_idx : ndarray – indices (not patient IDs) into the original array
        test_idx  : ndarray – indices (not patient IDs) into the original array
    """
    patient_indices = np.asarray(patient_indices)
    labels = np.asarray(labels)

    if num_splits >= 2:
        skf = StratifiedKFold(
            n_splits=num_splits,
            shuffle=True,
            random_state=seed,
        )
        for fold, (tr, te) in enumerate(skf.split(patient_indices, labels)):
            yield fold, patient_indices[tr], patient_indices[te]

    elif num_splits == 1:
        tr, te = train_test_split(
            patient_indices,
            test_size=test_size,
            stratify=labels,
            random_state=seed,
            shuffle=True,
        )
        yield 0, tr, te

    else:
        raise ValueError("num_splits must be >= 1")