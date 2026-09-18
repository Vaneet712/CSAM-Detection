# dataset/fold_loader.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


def parse_label(val) -> int:
    """Converts string/integer label variants to standard binary 0 (Non_CSAM) or 1 (CSAM)."""
    val_str = str(val).strip().lower()
    if val_str in ["0", "0.0", "non_csam", "noncsam", "safe", "normal", "benign"]:
        return 0
    elif val_str in ["1", "1.0", "csam", "csam_sens", "csam_non_sens", "flagged", "abusive", "harmful"]:
        return 1
    elif "non" in val_str or "safe" in val_str:
        return 0
    elif "csam" in val_str:
        return 1
    try:
        return int(float(val_str))
    except Exception:
        return 0


class CSVFoldDataset(Dataset):
    """
    Loads dataset directly from 5_fold_splits/fold_X/train.csv or test.csv
    """
    def __init__(self, csv_file: str, image_root: str = "dataset_updated_organized", transform=None):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        self.df = pd.read_csv(csv_file)
        self.image_root = image_root
        self.transform = transform
        self.samples = []

        # Find image and label column names
        col_map = {}
        for col in self.df.columns:
            clow = col.lower().strip()
            if clow in ["image", "image_path", "filename", "file_name", "path", "img"]:
                col_map[col] = "image"
            elif clow in ["label", "target", "class", "is_csam", "category"]:
                col_map[col] = "label"
        self.df.rename(columns=col_map, inplace=True)

        if "image" not in self.df.columns:
            self.df["image"] = self.df.iloc[:, 0]

        for _, row in self.df.iterrows():
            img_path = self._resolve_path(str(row["image"]))
            raw_label = row["label"] if "label" in self.df.columns else row["image"]
            label = parse_label(raw_label)
            self.samples.append((img_path, label))

    def _resolve_path(self, img_str: str) -> str:
        img_str = img_str.strip()
        if os.path.isabs(img_str) and os.path.exists(img_str):
            return img_str

        direct_path = os.path.join(self.image_root, img_str)
        if os.path.exists(direct_path):
            return direct_path

        base_name = os.path.basename(img_str)
        for root, _, files in os.walk(self.image_root):
            if base_name in files:
                return os.path.join(root, base_name)

        return direct_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), path