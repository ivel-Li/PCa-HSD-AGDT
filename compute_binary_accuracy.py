#!/usr/bin/env python3
"""
将4分类混淆矩阵合并为2类（良恶性），计算 (TP_benign + TP_cancer) / n_samples

合并规则:
  类别 0,1 → 良性 (benign)
  类别 2,3 → 恶性 (cancer)

用法:
    python compute_binary_accuracy.py                                 # 计算所有实验
    python compute_binary_accuracy.py run/LSDT-Swin-T/postmask/42     # 计算指定实验
"""

import os
import sys
import glob
import numpy as np
from collections import defaultdict


def compute_binary_metrics(cm: np.ndarray):
    """
    输入: cm (4x4 混淆矩阵, 行=真实标签, 列=预测标签)
    返回: binary_accuracy, TP_benign, TP_cancer, total
    """
    # TP_benign: 真实属于 {0,1} 且预测属于 {0,1}
    tp_benign = cm[0:2, 0:2].sum()
    # TP_cancer: 真实属于 {2,3} 且预测属于 {2,3}
    tp_cancer = cm[2:4, 2:4].sum()
    total = cm.sum()
    binary_acc = (tp_benign + tp_cancer) / total
    return binary_acc, tp_benign, tp_cancer, total


def main(target_dir="run"):
    if not os.path.isabs(target_dir):
        target_dir = os.path.abspath(target_dir)

    # 查找所有 confusion_matrix.npy 文件
    cm_files = sorted(glob.glob(os.path.join(target_dir, "**", "*confusion_matrix.npy"), recursive=True))

    if not cm_files:
        print(f"[ERROR] No confusion_matrix.npy files found under {target_dir}")
        sys.exit(1)

    # 按 (model, trick, seed) 分组
    experiments = defaultdict(list)
    for fpath in cm_files:
        parts = fpath.rstrip("/").split("/")
        # 从路径中提取 run/ 后的部分
        try:
            run_idx = parts.index("run")
            rel_parts = parts[run_idx + 1:]
        except ValueError:
            rel_parts = parts

        # 提取 fold_* 文件名
        fold_file = rel_parts[-1]
        fold_name = fold_file.replace("_confusion_matrix.npy", "")
        seed_name = rel_parts[-3]
        trick_num = rel_parts[-4]
        model_name = rel_parts[-5]

        experiments[(model_name, trick_num, seed_name)].append((fold_name, fpath))

    # 对每个实验计算
    results = []
    for (model_name, trick_num, seed_name), fold_list in sorted(experiments.items()):
        fold_list.sort(key=lambda x: x[0])  # 按 fold 排序

        per_fold_accs = []
        per_fold_details = []

        for fold_name, fpath in fold_list:
            cm = np.load(fpath)
            binary_acc, tp_benign, tp_cancer, total = compute_binary_metrics(cm)
            per_fold_accs.append(binary_acc)
            per_fold_details.append((fold_name, tp_benign, tp_cancer, total, binary_acc))

        mean_acc = np.mean(per_fold_accs)
        std_acc = np.std(per_fold_accs)

        results.append({
            "model": model_name,
            "trick": trick_num,
            "seed": seed_name,
            "per_fold": per_fold_details,
            "mean": mean_acc,
            "std": std_acc,
            "n_folds": len(per_fold_accs),
        })

    # 打印结果
    print(f"\n{'='*80}")
    print(f"  二元良恶性准确率  (TP_benign + TP_cancer) / n_samples")
    print(f"  合并规则: 类别 0,1 → 良性,  类别 2,3 → 恶性")
    print(f"{'='*80}")

    for r in results:
        print(f"\n  {r['model']} / {r['trick']} / seed={r['seed']}  ({r['n_folds']} folds):")
        for fn, tp_b, tp_c, total, acc in r["per_fold"]:
            print(f"    {fn}: (TP_b={int(tp_b):3d} + TP_c={int(tp_c):3d}) / {int(total):3d} = {acc*100:.2f}%  混淆矩阵: TP_b={int(tp_b)}, TP_c={int(tp_c)}, total={int(total)}")
        print(f"    Mean ± Std: {r['mean']*100:.2f}% ± {r['std']*100:.2f}%")

    # 按 (model, trick) 聚合
    print(f"\n{'='*80}")
    print(f"  按模型聚合 (跨种子)")
    print(f"{'='*80}")

    config_groups = defaultdict(list)
    for r in results:
        config_groups[(r["model"], r["trick"])].append(r)

    for (model_name, trick_num), group in config_groups.items():
        all_means = [r["mean"] for r in group]
        grand_mean = np.mean(all_means)
        grand_std = np.std(all_means)
        seeds = [r["seed"] for r in group]
        print(f"\n  {model_name} / {trick_num} ({len(group)} seeds: {', '.join(seeds)}):")
        print(f"    Per-seed binary accs: {[f'{m*100:.2f}%' for m in all_means]}")
        print(f"    Grand Mean ± Std:     {grand_mean*100:.2f}% ± {grand_std*100:.2f}%")

    # 汇总表
    print(f"\n{'='*80}")
    print(f"  汇总")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Trick':<12} {'Seed':<8} {'Binary Acc':<12} {'Std':<10} {'Folds'}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<20} {r['trick']:<12} {r['seed']:<8} {r['mean']*100:<10.2f}%  {r['std']*100:<8.2f}%  {r['n_folds']}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "run"
    main(target)