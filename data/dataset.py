"""
HDF5 Dataset loader for prostate cancer classification.
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from config import HDF5_PATH, DATA_EXTAND


class MRIDataset(Dataset):
    """
    Original notebook-style MRIDataset.
    Accepts a list of patient indices (after splitting) and loads data on-the-fly.
    Masks are loaded from 'T2_res_sam3_text_mask_0.6' by default (configurable via mask_key).

    When DATA_EXTAND=True (set in config), channels are expanded:
      - ADC channel gets concat of (ADC, ADC+DWI, T2)
    """

    def __init__(
        self,
        hdf5_path,
        patient_indices,
        use_rescaled=True,
        mask_key='T2_res_sam3_text_mask_0.6',
        load_mask=True,
    ):
        self.hdf5_path = hdf5_path
        self.patient_indices = patient_indices
        self.use_rescaled = use_rescaled
        self.mask_key = mask_key
        self.load_mask = load_mask
        self.file = None

    def __len__(self):
        return len(self.patient_indices)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, 'r')

        patient_index = self.patient_indices[idx]
        patient_group = self.file[str(patient_index)]

        label = patient_group['label'][()]
        item = {'label': torch.tensor(label, dtype=torch.long)}

        if self.use_rescaled:
            adc = patient_group['ADC'][:]  # (16, 224, 224, 1)
            dwi = patient_group['DWI'][:]
            t2 = patient_group['T2'][:]

            # Try loading mask via configurable self.mask_key; fallback to None if not found
            mask = None
            if self.load_mask:
                if self.mask_key in patient_group:
                    mask = patient_group[self.mask_key][:]
                elif 'mask_res' in patient_group:
                    mask = patient_group['mask_res'][:]

            item['adc'] = torch.from_numpy(adc).float().permute(0, 3, 1, 2)
            item['dwi'] = torch.from_numpy(dwi).float().permute(0, 3, 1, 2)
            item['t2'] = torch.from_numpy(t2).float().permute(0, 3, 1, 2)
            if mask is not None:
                item['mask'] = torch.from_numpy(mask).float().permute(0, 3, 1, 2)
        else:
            adc = patient_group['ADC'][:]  # (16, 224, 224, 1)
            dwi = patient_group['DWI'][:]
            t2 = patient_group['T2'][:]

            item['adc'] = torch.from_numpy(adc).float().permute(0, 3, 1, 2)
            item['dwi'] = torch.from_numpy(dwi).float().permute(0, 3, 1, 2)
            item['t2'] = torch.from_numpy(t2).float().permute(0, 3, 1, 2)

        # DATA_EXTAND: expand ADC channel dimension
        if DATA_EXTAND:
            # Replicate notebook logic: cat adc, (adc+dwi), t2 along channel dim
            adc_dwi = item['adc'] + item['dwi']
            t2_rep = item['t2'].repeat(1, 2, 1, 1)  # (S, 2, H, W)
            item['adc'] = torch.cat([item['adc'], adc_dwi, t2_rep], dim=1)  # (S, 1+1+2=4, H, W)
            # Also expand dwi and t2 to match channel count for uniform model input
            item['dwi'] = item['dwi'].repeat(1, 4, 1, 1)
            item['t2'] = item['t2'].repeat(1, 4, 1, 1)

        return item


class ProstateMRIDataset(Dataset):
    """
    Alternative dataset that loads data by patient ID string.
    Useful when you want the dataset to manage its own train/test split.
    """

    def __init__(self, hdf5_path=None, transform=None, split='train', test_size=0.2):
        self.hdf5_path = hdf5_path or HDF5_PATH
        self.transform = transform

        with h5py.File(self.hdf5_path, 'r') as f:
            self.patient_ids = list(f.keys())

        # Split
        np.random.seed(42)
        n = len(self.patient_ids)
        indices = np.random.permutation(n)
        split_idx = int(n * (1 - test_size))
        if split == 'train':
            self.patient_ids = [self.patient_ids[i] for i in indices[:split_idx]]
        else:
            self.patient_ids = [self.patient_ids[i] for i in indices[split_idx:]]

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        with h5py.File(self.hdf5_path, 'r') as f:
            group = f[pid]
            label = int(group['label'][()])

            adc = group['ADC_res'][:]
            dwi = group['DWI_res'][:]
            t2 = group['T2_res'][:]

            adc = torch.tensor(adc, dtype=torch.float32).permute(0, 3, 1, 2)
            dwi = torch.tensor(dwi, dtype=torch.float32).permute(0, 3, 1, 2)
            t2 = torch.tensor(t2, dtype=torch.float32).permute(0, 3, 1, 2)

            try:
                mask = group['mask_res'][:]
                mask = torch.tensor(mask, dtype=torch.float32).permute(0, 3, 1, 2)
            except (KeyError, ValueError):
                mask = None

        if DATA_EXTAND:
            adc_dwi = torch.cat([adc, dwi], dim=1)
            t2_rep = t2.repeat(1, 2, 1, 1)
            adc = torch.cat([adc, adc_dwi, t2_rep], dim=1)

        return {
            'adc': adc,
            'dwi': dwi,
            't2': t2,
            'label': label,
            'mask': mask,
        }
