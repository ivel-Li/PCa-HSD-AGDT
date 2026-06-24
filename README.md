# Histopathological Spectrum-Guided Prostate Stratification via Segmentation-Assisted Diagnostic Transformer

Official repository for the paper *"Histopathological Spectrum-Guided Prostate Stratification via Segmentation-Assisted Diagnostic Transformer"*.

**Contributions:**
- **Pathology-grounded benchmark** — A four-class prostate mpMRI dataset with biopsy-confirmed labels, enabling clinically meaningful imaging–pathology research.
- **Segmentation-assisted tri-modal 3D classification network (LSDT)** — Incorporates SAM3 foundation model priors and integrates slice sequential features from T2WI, ADC, and DWI for fine-grained classification.
- **Cross-institutional generalization** — Strong results on both our benchmark and the public PI-CAI dataset, suggesting clinical utility for risk stratification.

> 🔗 PI-CAI baseline: [github.com/ivel-Li/picai_baseline_classification](https://github.com/ivel-Li/picai_baseline_classification)

---

## Pipeline Overview

```
Raw DICOM/NIfTI (T2WI, ADC, DWI)
  │
  ├── Registration & Normalization
  │     (rigid registration ADC/DWI → T2WI; resample to isotropic; intensity norm)
  │
  ├── HDF5 Packaging ── dataset/2Input_v1.1_no1024.ipynb
  │     Extract 16 center slices → resize to 224×224 → min-max norm [0,1]
  │     ➜ dataset/patients_dataset_v1.0.h5
  │       (344 patients, each: ADC, DWI, T2, label)
  │
  ├── SAM3 Mask Generation ── dataset/MaskProcess.py
  │     T2WI per slice → SAM3 text prompt "prostate" (threshold 0.6)
  │     ➜ /<patient>/T2_res_sam3_text_mask_0.6
  │     (SAM3 repo: https://github.com/ivel-Li/sam3)
  │
  └── Training & Evaluation ── python main.py
```

### Label Mapping

| Label | Class |
|-------|-------|
| 0 | Normal prostate |
| 1 | Benign prostate cancer |
| 2 | Non-significant prostate cancer |
| 3 | Significant prostate cancer |

Two dataset versions: `patients_dataset_v1.0.h5` (344 patients) and `patients_dataset_v1.0_no1024.h5` (removed patient #1024, low quality).

### Model Architecture

![Model Architecture Overview](overview.png)

*Figure 1: Language-guided Segmentation-assisted Diagnostic Transformer (LSDT).*

The LSDT framework:
1. **Tri-modal input** — 16 axial slices × 3 modalities (T2WI, ADC, DWI) → (3 × 16 × 224 × 224).
2. **SAM3 prior mask** — Binary prostate mask (16 × 224 × 224) guides attention to the ROI.
3. **Feature extraction** — Shared backbone (ViT/Swin/ResNet/BiomedCLIP/OmniRad).
4. **Mask-guided fusion** — Transformer/cat/CNN fusion of multi-modal features.
5. **Slice sequential aggregation** — 16 slice features → patient-level representation.
6. **Classification head** — 4-class output.

---

## Supported Models

### LSDT (with SAM3 mask + Transformer fusion)

| Name | Backbone |
|------|----------|
| `LSDT-ResNet50` | ResNet50 |
| `LSDT-Swin-T` | Swin Base |
| `LSDT-Base` | ViT Base/16 |
| `LSDT-Large` | ViT Large/16 |
| `LSDT-BiomedCLIP` | BiomedCLIP ViT-B/16 |
| `LSDT-Omnirad` | OmniRad ViT-B/16 |

### Standard Classifiers (no mask)

`ResNetClassifier`, `MRIClassifier`, `VITClassifier`, `VITLargeClassifier`, `VITLarge21kClassifier`, `VITHugeClassifier`, `SWINClassifier`, `Biomedclip_vit`, `Omnirad_vit`

> Pretrained weights go in `./weights/`; falls back to random init if absent.

---

## Quick Start

### 1. Setup & Dataset Preprocessing

```bash
# ── Step A: Create environment and install dependencies ────
conda create -n PCa-HSD python=3.10
conda activate PCa-HSD
pip install -r requirements.txt

# ── Step B: Organize raw data ─────────────────────────────
# Place NIfTI volumes under dataset/v1.0/ 

# ── Step C: Build HDF5 dataset ────────────────────────────
python dataset/build_h5_dataset.py

# ── Step D (optional): Generate SAM3 prostate masks ───────
# Install SAM3 for mask generation: https://github.com/facebookresearch/sam3.git
python dataset/MaskProcess.py
```

### 2. Training & Evaluation

```bash
conda activate PCa-HSD

# Default run (LSDT-Large with SAM3 mask)
python main.py

# Custom config / GPU / Seed
python main.py --config config_resnet
python main.py --gpu cuda:1
python main.py --seed 123                    # Override seed (default: 42)
python main.py --config config_lsdt_base --seed 42 --seed 123  # Multi-seed runs
```

### Key Config (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `LSDT-Large` | Model architecture |
| `num_epochs` | 150 | Training epochs |
| `batch_size` | 16 | Batch size |
| `num_splits` | 5 | K-fold folds |
| `DATA_EXTAND` | `False` | Concatenate ADC+DWI channels |
| `SupCon` | `False` | Supervised contrastive loss |
| `HDF5_PATH` | `"./dataset/patients_dataset_v1.0.h5"` | Dataset path |

---

## Project Structure

```
PCa-HSD-LSDT/
├── main.py / config.py / train.py     # Entry, config, training loop
├── data/
│   ├── dataset.py                     # MRIDataset — HDF5 loader
│   └── split.py                       # Stratified K-fold split
├── models/
│   ├── model_factory.py               # build_model() registry
│   ├── base.py / resnet.py / vit.py / bio_vit.py
├── utils/losses.py                    # SupConLoss
├── dataset/
│   ├── 2Input_v1.1[_no1024].ipynb     # HDF5 builders
│   ├── MaskProcess.py                 # SAM3 mask generation
│   ├── patients_dataset_v1.0[_no1024].h5
│   └── v1.0/ v1.1/                   # Raw NIfTI volumes
├── weights/                           # Pretrained checkpoints
└── run/                               # Output directory
```

---

## Reproducing Paper Results

```bash
python main.py                              # LSDT-Omnirad (default, seed=42)
python main.py --config config_lsdt_base    # LSDT-Base
python main.py --config config_omnirad_base  # Ablation: no SAM3 mask
python main.py --config config_lsdt_t2_only  # Ablation: T2WI only

# Multi-seed experiments (e.g. seeds 42, 123, 456)
python main.py --seed 42
python main.py --seed 123
python main.py --seed 456
```

---

## PI-CAI Benchmark

Evaluate cross-institutional generalization by pointing `HDF5_PATH` to the PI-CAI HDF5 file (see [preprocessing repo](https://github.com/ivel-Li/picai_baseline_classification)).

---

## Output

```
{run}/fold_{i}/
├── {run}_best_model.pth
├── {run}_training_log.json
├── Tra_Val_Loss.png / Tra_Val_Acc.png
└── fold_{i}_confusion_matrix.png
```

---

## Citation

```bibtex
@article{yourcitation2024,
  title={Histopathological Spectrum-Guided Prostate Stratification via Segmentation-Assisted Diagnostic Transformer},
  author={Your Authors},
  journal={Under Review},
  year={2024}
}
```

---

## License & Acknowledgements

- MIT License
- [PI-CAI Baseline](https://github.com/DIAGNijmegen/picai_baseline) — Public benchmark and baseline implementation
- [SAM3](https://github.com/facebookresearch/sam3) — Foundation model for medical image segmentation
- [PI-CAI](https://pi-cai.grand-challenge.org/) — Challenge and dataset
