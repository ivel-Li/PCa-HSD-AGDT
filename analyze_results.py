#!/usr/bin/env python3
"""
Analyze all experiment results under run/.
For each configuration (model + trick + seed):
  - Reads each fold's training_log.json
  - Gets best val_acc (max test_acc) per fold
  - Computes mean ± std across 5 folds
  - Shows per-fold best accuracies

Usage:
    python analyze_results.py                       # analyze all
    python analyze_results.py run/VITLargeClassifier # analyze specific path
"""

import os
import sys
import json
import glob
import numpy as np
from collections import OrderedDict


def get_best_acc_from_log(log_path):
    """Return max test_acc, corresponding epoch, and associated joint_recall from a training log JSON."""
    with open(log_path) as f:
        data = json.load(f)
    accs = data["test_acc"]
    best_idx = int(np.argmax(accs))
    best = accs[best_idx]
    best_epoch = data["epoch"][best_idx]
    # Joint recall at best acc epoch (if available in log)
    jr = None
    if "test_joint_recall" in data and best_idx < len(data["test_joint_recall"]):
        jr = data["test_joint_recall"][best_idx]
    return best, best_epoch, jr


def get_best_jr_from_log(log_path):
    """Return max joint_recall and corresponding epoch and accuracy."""
    with open(log_path) as f:
        data = json.load(f)
    if "test_joint_recall" not in data or len(data["test_joint_recall"]) == 0:
        return None, None, None
    jrs = data["test_joint_recall"]
    best_idx = int(np.argmax(jrs))
    best_jr = jrs[best_idx]
    best_epoch = data["epoch"][best_idx]
    acc_at_best_jr = data["test_acc"][best_idx]
    return best_jr, best_epoch, acc_at_best_jr


def analyze_experiment(exp_path):
    """
    Analyze a single experiment directory (e.g., run/LSDT-Large/5/seed42).
    Returns a dict with best_accs (per fold), best_jrs (per fold), mean, std, etc.
    """
    # Find all fold directories
    fold_dirs = sorted([
        d for d in os.listdir(exp_path)
        if d.startswith("fold_") and os.path.isdir(os.path.join(exp_path, d))
    ])

    if not fold_dirs:
        return None

    best_accs = []
    best_epochs = []
    best_jrs_at_best_acc = []
    best_jrs = []
    best_jr_epochs = []
    accs_at_best_jr = []
    fold_names = []

    for fd in fold_dirs:
        fold_path = os.path.join(exp_path, fd)
        # Find training_log.json (naming varies: 42_training_log.json, seed42_training_log.json, etc.)
        log_files = glob.glob(os.path.join(fold_path, "*_training_log.json"))
        if not log_files:
            print(f"  [WARN] No training log found in {fold_path}")
            continue

        best_acc, best_epoch, jr_at_best_acc = get_best_acc_from_log(log_files[0])
        best_jr, best_jr_epoch, acc_at_best_jr_val = get_best_jr_from_log(log_files[0])

        best_accs.append(best_acc)
        best_epochs.append(best_epoch)
        best_jrs_at_best_acc.append(jr_at_best_acc)
        best_jrs.append(best_jr)
        best_jr_epochs.append(best_jr_epoch)
        accs_at_best_jr.append(acc_at_best_jr_val)
        fold_names.append(fd)

    if len(best_accs) == 0:
        return None

    return {
        "fold_names": fold_names,
        "best_accs": best_accs,
        "best_epochs": best_epochs,
        "best_jrs_at_best_acc": best_jrs_at_best_acc,
        "best_jrs": best_jrs,
        "best_jr_epochs": best_jr_epochs,
        "accs_at_best_jr": accs_at_best_jr,
        "mean": np.mean(best_accs),
        "std": np.std(best_accs),
        "mean_jr": np.mean([j for j in best_jrs_at_best_acc if j is not None]) if any(j is not None for j in best_jrs_at_best_acc) else None,
        "n_folds": len(best_accs),
    }


def print_header(text):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def main(root_dir="run"):
    # Convert to absolute path
    if not os.path.isabs(root_dir):
        root_dir = os.path.abspath(root_dir)

    if not os.path.exists(root_dir):
        print(f"[ERROR] Path not found: {root_dir}")
        sys.exit(1)

    # Gather all experiment directories (leaf dirs containing fold_* dirs)
    # Strategy: walk down finding dirs that have subdirs like fold_0, fold_1, etc.
    experiment_dirs = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        # Check if this dir contains fold_0, fold_1, etc.
        fold_dirs = [d for d in dirnames if d.startswith("fold_")]
        if len(fold_dirs) >= 3:  # at least 3 folds to be considered an exp dir
            experiment_dirs.append(dirpath)

    experiment_dirs.sort()

    if not experiment_dirs:
        print(f"[ERROR] No experiment directories found under {root_dir}")
        print("  (Expected directories containing fold_0, fold_1, ...)")
        sys.exit(1)

    # Organize by model → trick_num → seed
    # Path pattern: run/{model}/{trick_num}/{seed}
    experiments_by_config = OrderedDict()

    for exp_dir in experiment_dirs:
        parts = exp_dir.rstrip("/").split("/")
        # parts: ['run', model, trick_num, seed] or deeper
        # Find the relevant parts
        try:
            run_idx = parts.index("run")
            rel_parts = parts[run_idx+1:]  # relative to run/
        except ValueError:
            rel_parts = parts

        if len(rel_parts) >= 3:
            model_name = rel_parts[-3]
            trick_num = rel_parts[-2]
            seed_name = rel_parts[-1]
        elif len(rel_parts) == 2:
            model_name = rel_parts[-2]
            trick_num = rel_parts[-1]
            seed_name = ""
        else:
            model_name = rel_parts[-1]
            trick_num = ""
            seed_name = ""

        key = (model_name, trick_num)
        if key not in experiments_by_config:
            experiments_by_config[key] = []
        experiments_by_config[key].append((seed_name, exp_dir))

    # Analyze each config group
    all_results = []

    for (model_name, trick_num), seed_list in experiments_by_config.items():
        print_header(f"{model_name} / trick={trick_num}")

        for seed_name, exp_dir in seed_list:
            result = analyze_experiment(exp_dir)
            if result is None:
                print(f"  {seed_name}: no results")
                continue

            # Format per-fold lines
            acc_str = "  ".join([
                f"{fn}={acc:.2f}%"
                for fn, acc in zip(result["fold_names"], result["best_accs"])
            ])
            jr_at_acc_str = "  ".join([
                f"{j:.4f}" if j is not None else "N/A"
                for j in result["best_jrs_at_best_acc"]
            ])
            best_jr_str = "  ".join([
                f"{j:.4f}" if j is not None else "N/A"
                for j in result["best_jrs"]
            ])
            acc_at_best_jr_str = "  ".join([
                f"{a:.2f}%" if a is not None else "N/A"
                for a in result["accs_at_best_jr"]
            ])
            # JR @ best acc: mean and std
            jr_at_best_acc_vals = [j for j in result["best_jrs_at_best_acc"] if j is not None]
            mean_jr_at_best_acc = np.mean(jr_at_best_acc_vals) if jr_at_best_acc_vals else None
            std_jr_at_best_acc = np.std(jr_at_best_acc_vals) if jr_at_best_acc_vals else None
            # Mean of Best JR
            best_jr_vals = [j for j in result["best_jrs"] if j is not None]
            mean_best_jr = np.mean(best_jr_vals) if best_jr_vals else None

            print(f"\n  Seed {seed_name}:")
            print(f"    Per-fold best acc:  {acc_str}")
            print(f"    Mean ± Std:         {result['mean']:.2f}% ± {result['std']:.2f}%")
            print(f"    Best epochs:        {result['best_epochs']}")
            print(f"    JR @ best acc:      {jr_at_acc_str}")
            if mean_jr_at_best_acc is not None:
                print(f"    Mean JR @ best acc: {mean_jr_at_best_acc:.4f} ± {std_jr_at_best_acc:.4f}")
            if mean_best_jr is not None:
                print(f"    Best JR per fold:   {best_jr_str}")
                print(f"    Acc @ best JR:      {acc_at_best_jr_str}")
                print(f"    Mean Best JR:       {mean_best_jr:.4f}")

            all_results.append({
                "model": model_name,
                "trick": trick_num,
                "seed": seed_name,
                "path": exp_dir,
                "per_fold_acc": result["best_accs"],
                "mean_acc": round(result["mean"], 2),
                "std_acc": round(result["std"], 2),
                "best_epochs": result["best_epochs"],
                "per_fold_jr": result["best_jrs"],
                "mean_jr": round(result["mean_jr"], 4) if result["mean_jr"] is not None else None,
            })

    # Print summary table
    print_header("SUMMARY TABLE")
    print(f"{'Model':<20} {'Trick':<12} {'Seed':<10} {'Mean Acc':<10} {'Std Acc':<10} {'Mean JR':<10} {'Folds':<8}")
    print("-" * 85)
    for r in all_results:
        n_folds = len(r["per_fold_acc"])
        jr_str = f"{r['mean_jr']:.4f}" if r['mean_jr'] is not None else "N/A"
        print(f"{r['model']:<20} {r['trick']:<12} {r['seed']:<10} "
              f"{r['mean_acc']:<8.2f}%  {r['std_acc']:<8.2f}%  {jr_str:<10} {n_folds}")

    # Per-config aggregation (mean of seeds)
    print_header("PER-CONFIG AGGREGATION (across seeds)")
    from collections import defaultdict
    config_groups = defaultdict(list)
    for r in all_results:
        config_groups[(r["model"], r["trick"])].append(r)

    for (model_name, trick_num), results in config_groups.items():
        all_seed_means = [r["mean_acc"] for r in results]
        all_seed_stds = [r["std_acc"] for r in results]
        all_seed_jrs = [r["mean_jr"] for r in results if r["mean_jr"] is not None]
        print(f"\n  {model_name} / {trick_num} ({len(results)} seeds):")
        print(f"    Per-seed means:     {[f'{m:.2f}%' for m in all_seed_means]}")
        print(f"    Grand Mean ± Std:   {np.mean(all_seed_means):.2f}% ± {np.std(all_seed_means):.2f}%")
        if all_seed_jrs:
            print(f"    Per-seed JR means:  {[f'{j:.4f}' for j in all_seed_jrs]}")
            print(f"    Grand JR Mean ± Std: {np.mean(all_seed_jrs):.4f} ± {np.std(all_seed_jrs):.4f}")

    if all(r["mean_jr"] is None for r in all_results):
        print_header("NOTE: Cancer Joint Recall")
        print("  The training_log.json does NOT contain joint recall values in these runs.")
        print("  The updated training pipeline (train.py + main.py) now logs test_joint_recall")
        print("  during validation, so future runs will have it.")
        print("")
        print("  For existing runs, use utils/summarize.py to recompute:")
        print("    python utils/summarize.py run/LSDT-Large/5/seed42")
        print("    for d in run/LSDT-Large/5/seed*/; do python utils/summarize.py \"$d\"; done")


if __name__ == "__main__":
    # Optional: pass a specific path to analyze
    target = sys.argv[1] if len(sys.argv) > 1 else "run"
    main(target)
