"""
Bridge crack inspection dataset with robust augmentation pipeline.
Supports SDNET2018 format: {root}/{D,P,W}/{C*,U*}/*.jpg
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


def get_robust_train_transform(img_size: int = 224) -> A.Compose:
    """Inspection-scene robust augmentation simulating real UAV degradation."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        # --- directional degradation simulation ---
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.5),
        A.MotionBlur(blur_limit=(3, 7), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.2),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1,
                       num_shadows_upper=3, shadow_dimension=5, p=0.25),
        A.Perspective(scale=(0.03, 0.08), p=0.2),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08, p=0.3),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
        A.CLAHE(clip_limit=2.0, p=0.15),
        # --- normalize & tensor ---
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ])


def get_standard_train_transform(img_size: int = 224) -> A.Compose:
    """Standard augmentation baseline (no inspection-specific degradation)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ])


def get_val_transform(img_size: int = 224) -> A.Compose:
    """Validation/test transform (no augmentation)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ])


class CrackDataset(Dataset):
    """Binary classification dataset: crack (1) vs non-crack (0)."""

    def __init__(self, file_list: List[Tuple[str, int]], transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

    def get_labels(self):
        return [lbl for _, lbl in self.file_list]


def collect_sdnet_files(data_root: str, categories: Tuple[str, ...] = ("D",)) -> List[Tuple[str, int]]:
    """
    Collect image paths and labels from SDNET2018 layout.
    categories: subset of ('D', 'P', 'W')  –  D=Deck, P=Pavement, W=Wall
    Returns list of (filepath, label) where label 1=crack, 0=non-crack.
    """
    samples = []
    for cat in categories:
        crack_dir = os.path.join(data_root, cat, f"C{cat}")
        nocrack_dir = os.path.join(data_root, cat, f"U{cat}")
        if os.path.isdir(crack_dir):
            for f in os.listdir(crack_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    samples.append((os.path.join(crack_dir, f), 1))
        if os.path.isdir(nocrack_dir):
            for f in os.listdir(nocrack_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    samples.append((os.path.join(nocrack_dir, f), 0))
    return samples


def split_dataset(
    samples: List[Tuple[str, int]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """Stratified split into train / val / test."""
    random.seed(seed)
    crack = [s for s in samples if s[1] == 1]
    nocrack = [s for s in samples if s[1] == 0]
    random.shuffle(crack)
    random.shuffle(nocrack)

    def _split(lst):
        n = len(lst)
        t1 = int(n * train_ratio)
        t2 = int(n * (train_ratio + val_ratio))
        return lst[:t1], lst[t1:t2], lst[t2:]

    c_tr, c_val, c_te = _split(crack)
    n_tr, n_val, n_te = _split(nocrack)
    train = c_tr + n_tr
    val = c_val + n_val
    test = c_te + n_te
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def split_dataset_kfold(
    samples: List[Tuple[str, int]],
    fold: int,
    n_folds: int = 5,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Stratified K-fold: one fold held out as test (~1/n_folds);
    remaining pool split into train / val stratified such that validation
    fraction of *total* dataset ≈ original 70/15/15 design (here val ≈ 15% of total).
    """
    if fold < 0 or fold >= n_folds:
        raise ValueError(f"fold must be in [0, {n_folds - 1}], got {fold}")
    if len(samples) < n_folds * 2:
        raise ValueError("Not enough samples for stratified K-fold.")

    indices = np.arange(len(samples))
    labels = np.array([s[1] for s in samples], dtype=int)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(indices, labels))
    train_val_idx, test_idx = splits[fold]

    tv_labels = labels[train_val_idx]
    # Aim for val ~ 15% of all samples ⇒ val / train_val_pool = 15% / (100% − 20%) = 15/80
    val_frac_in_pool = 15.0 / 80.0
    rel_tr, rel_val = train_test_split(
        np.arange(len(train_val_idx)),
        test_size=val_frac_in_pool,
        stratify=tv_labels,
        random_state=seed + fold * 9973,
    )
    ti_tr = train_val_idx[rel_tr]
    ti_va = train_val_idx[rel_val]

    train_list = [samples[i] for i in ti_tr]
    val_list = [samples[i] for i in ti_va]
    test_list = [samples[i] for i in test_idx]

    random.seed(seed + fold)
    random.shuffle(train_list)
    random.shuffle(val_list)
    random.shuffle(test_list)
    return train_list, val_list, test_list


def make_weighted_sampler(dataset: CrackDataset) -> WeightedRandomSampler:
    """Create a weighted sampler to address class imbalance during training."""
    labels = dataset.get_labels()
    class_counts = np.bincount(labels)
    weights_per_class = 1.0 / class_counts.astype(np.float64)
    sample_weights = [weights_per_class[lbl] for lbl in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def build_dataloaders(
    data_root: str,
    categories: Tuple[str, ...] = ("D",),
    img_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 8,
    use_robust_aug: bool = True,
    use_weighted_sampler: bool = True,
    seed: int = 42,
    fold: Optional[int] = None,
    n_folds: int = 5,
    cv_seed: int = 42,
) -> Dict[str, DataLoader]:
    """Build train/val/test dataloaders.

    When ``fold`` is not None: stratified ``n_folds`` CV; this run uses fold ``fold``
    test split (same folds for all configs if ``cv_seed`` is fixed).
    """
    samples = collect_sdnet_files(data_root, categories)
    if fold is None:
        train_list, val_list, test_list = split_dataset(samples, seed=seed)
    else:
        train_list, val_list, test_list = split_dataset_kfold(
            samples, fold=fold, n_folds=n_folds, seed=cv_seed
        )

    train_tf = get_robust_train_transform(img_size) if use_robust_aug else get_standard_train_transform(img_size)
    val_tf = get_val_transform(img_size)

    train_ds = CrackDataset(train_list, transform=train_tf)
    val_ds = CrackDataset(val_list, transform=val_tf)
    test_ds = CrackDataset(test_list, transform=val_tf)

    sampler = make_weighted_sampler(train_ds) if use_weighted_sampler else None
    shuffle_train = (sampler is None)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                            shuffle=shuffle_train, num_workers=num_workers,
                            pin_memory=True, drop_last=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True),
    }

    labels_arr = np.array(train_ds.get_labels(), dtype=np.int64)
    counts = np.bincount(labels_arr, minlength=2).astype(np.float64)
    inv_freq = len(labels_arr) / (2.0 * np.maximum(counts, 1.0))
    class_weights = torch.tensor(inv_freq, dtype=torch.float32)

    info = {
        "train_total": len(train_ds),
        "val_total": len(val_ds),
        "test_total": len(test_ds),
        "train_crack": sum(1 for _, l in train_list if l == 1),
        "train_nocrack": sum(1 for _, l in train_list if l == 0),
        "class_counts": counts.tolist(),
        "class_weights": class_weights,
    }
    cv_tag = f" CV fold={fold}/{n_folds}" if fold is not None else ""
    print(f"[Dataset]{cv_tag} Train: {info['train_total']} (crack={info['train_crack']}, "
          f"nocrack={info['train_nocrack']}) | Val: {info['val_total']} | Test: {info['test_total']}")
    return loaders, info
