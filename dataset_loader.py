"""
Dataset Loader for Multimodal CSAM Baseline Evaluation
======================================================
Supports:
- Image loading (PNG, JPG, JPEG, WEBP, BMP, etc.)
- Video frame directory loading (e.g., video012/frame_*.jpg)
- Auto-discovery across splits: train, val, test
- Auto-discovery across classes: CSAM_non_sens, CSAM_sens, Non_CSAM
- Auto-discovery and parsing of split CSVs (test_csam_non_sens.csv, test_non_csam.csv, metadata.csv, etc.)
- Binary label mapping (CSAM vs Non-CSAM) and 3-Class label mapping
- Uniform frame sampling for video temporal consistency
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

CLASS_MAP_3CLASS = {
    'CSAM_non_sens': 0,
    'CSAM_sens': 1,
    'Non_CSAM': 2
}

CLASS_MAP_BINARY = {
    'CSAM_non_sens': 1,
    'CSAM_sens': 1,
    'Non_CSAM': 0
}

CLASS_NAMES_3CLASS = ['CSAM_non_sens', 'CSAM_sens', 'Non_CSAM']
CLASS_NAMES_BINARY = ['Non_CSAM', 'CSAM']


class CSAMDataset(Dataset):
    """
    Unified Multimodal Dataset for Images and Video Frame Directories.
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'test',
        classification_mode: str = 'binary',  # 'binary' or '3class'
        max_video_frames: int = 8,
        transform=None,
        metadata_csv: Optional[str] = None
    ):
        """
        Args:
            root_dir: Path to `dataset_updated_organized` or dataset root
            split: 'test', 'val', 'train', or 'all'
            classification_mode: 'binary' (CSAM vs Non-CSAM) or '3class' (CSAM_non_sens, CSAM_sens, Non_CSAM)
            max_video_frames: Number of uniform frames to sample per video instance
            transform: Optional torchvision transform for images/frames
            metadata_csv: Optional explicit path to metadata.csv
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.classification_mode = classification_mode
        self.max_video_frames = max_video_frames
        self.transform = transform
        self.samples: List[Dict[str, Any]] = []

        self._discover_dataset(metadata_csv)

    def _discover_dataset(self, metadata_csv: Optional[str] = None):
        """Discovers all images and video frame folders in the dataset via directory structure and CSVs."""
        splits_to_scan = [self.split] if self.split != 'all' else ['train', 'val', 'test']
        seen_paths = set()

        for s in splits_to_scan:
            split_path = self.root_dir / s
            if not split_path.exists():
                # Maybe root_dir itself contains class folders or split_path is root_dir
                split_path = self.root_dir

            # 1. Check for split-specific CSVs if available
            csv_candidates = [
                split_path / f"{s}_csam_non_sens.csv",
                split_path / f"{s}_non_csam.csv",
                split_path / f"{s}_csam_sens.csv",
                split_path / f"{s}_csam.csv",
                split_path / "metadata.csv",
                self.root_dir / f"{s}_csam_non_sens.csv",
                self.root_dir / f"{s}_non_csam.csv",
                self.root_dir / "metadata.csv"
            ]
            if metadata_csv:
                csv_candidates.insert(0, Path(metadata_csv))

            csv_samples_loaded = 0
            for csv_file in csv_candidates:
                if csv_file.exists() and csv_file.is_file():
                    try:
                        df = pd.read_csv(csv_file)
                        inferred_class = None
                        if 'csam_non_sens' in csv_file.name.lower():
                            inferred_class = 'CSAM_non_sens'
                        elif 'csam_sens' in csv_file.name.lower():
                            inferred_class = 'CSAM_sens'
                        elif 'non_csam' in csv_file.name.lower():
                            inferred_class = 'Non_CSAM'

                        for _, row in df.iterrows():
                            path_val = None
                            for col in ['path', 'file_path', 'filepath', 'image_path', 'video_path', 'file', 'image', 'video', 'filename']:
                                if col in row and pd.notna(row[col]):
                                    path_val = str(row[col])
                                    break

                            if not path_val:
                                continue

                            cand_p = Path(path_val)
                            if not cand_p.is_absolute():
                                if (split_path / cand_p).exists():
                                    full_p = split_path / cand_p
                                elif (self.root_dir / cand_p).exists():
                                    full_p = self.root_dir / cand_p
                                else:
                                    full_p = cand_p
                            else:
                                full_p = cand_p

                            if str(full_p) in seen_paths:
                                continue

                            row_class = None
                            for col in ['class', 'class_name', 'label_name', 'category']:
                                if col in row and pd.notna(row[col]):
                                    row_class = str(row[col])
                                    break
                            class_name = row_class or inferred_class or 'Non_CSAM'
                            if class_name not in CLASS_MAP_BINARY:
                                if 'non_sens' in class_name.lower():
                                    class_name = 'CSAM_non_sens'
                                elif 'sens' in class_name.lower() and 'non' not in class_name.lower():
                                    class_name = 'CSAM_sens'
                                else:
                                    class_name = 'Non_CSAM'

                            binary_label = CLASS_MAP_BINARY.get(class_name, 0)
                            label_3class = CLASS_MAP_3CLASS.get(class_name, 2)
                            target_label = binary_label if self.classification_mode == 'binary' else label_3class

                            if full_p.is_dir():
                                frame_files = sorted([
                                    str(f) for f in full_p.iterdir()
                                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                                ])
                                if len(frame_files) > 0:
                                    seen_paths.add(str(full_p))
                                    self.samples.append({
                                        'id': f"{s}_{class_name}_vid_{full_p.name}",
                                        'path': str(full_p),
                                        'type': 'video',
                                        'class_name': class_name,
                                        'binary_label': binary_label,
                                        'label_3class': label_3class,
                                        'label': target_label,
                                        'split': s,
                                        'num_frames': len(frame_files),
                                        'frame_paths': frame_files
                                    })
                                    csv_samples_loaded += 1
                            elif full_p.is_file() and full_p.suffix.lower() in IMAGE_EXTENSIONS:
                                seen_paths.add(str(full_p))
                                self.samples.append({
                                    'id': f"{s}_{class_name}_img_{full_p.stem}",
                                    'path': str(full_p),
                                    'type': 'image',
                                    'class_name': class_name,
                                    'binary_label': binary_label,
                                    'label_3class': label_3class,
                                    'label': target_label,
                                    'split': s,
                                    'num_frames': 1,
                                    'frame_paths': [str(full_p)]
                                })
                                csv_samples_loaded += 1
                    except Exception:
                        pass

            # 2. Directory structure scan: split_path / class_name / images & videos
            for class_name in ['CSAM_non_sens', 'CSAM_sens', 'Non_CSAM']:
                class_path = split_path / class_name
                if not class_path.exists():
                    for sub in split_path.glob('*'):
                        if sub.is_dir() and sub.name.lower() == class_name.lower():
                            class_path = sub
                            break

                if not class_path.exists():
                    continue

                binary_label = CLASS_MAP_BINARY.get(class_name, 0)
                label_3class = CLASS_MAP_3CLASS.get(class_name, 2)
                target_label = binary_label if self.classification_mode == 'binary' else label_3class

                # (a) Discover Images
                images_dir = class_path / 'images'
                if images_dir.exists():
                    for img_file in images_dir.iterdir():
                        if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
                            if str(img_file) not in seen_paths:
                                seen_paths.add(str(img_file))
                                self.samples.append({
                                    'id': f"{s}_{class_name}_img_{img_file.stem}",
                                    'path': str(img_file),
                                    'type': 'image',
                                    'class_name': class_name,
                                    'binary_label': binary_label,
                                    'label_3class': label_3class,
                                    'label': target_label,
                                    'split': s,
                                    'num_frames': 1,
                                    'frame_paths': [str(img_file)]
                                })

                # (b) Discover Videos (Folders of frames)
                videos_dir = class_path / 'videos'
                if videos_dir.exists():
                    for video_folder in videos_dir.iterdir():
                        if video_folder.is_dir():
                            if str(video_folder) not in seen_paths:
                                frame_files = sorted([
                                    str(f) for f in video_folder.iterdir()
                                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                                ])
                                if len(frame_files) > 0:
                                    seen_paths.add(str(video_folder))
                                    self.samples.append({
                                        'id': f"{s}_{class_name}_vid_{video_folder.name}",
                                        'path': str(video_folder),
                                        'type': 'video',
                                        'class_name': class_name,
                                        'binary_label': binary_label,
                                        'label_3class': label_3class,
                                        'label': target_label,
                                        'split': s,
                                        'num_frames': len(frame_files),
                                        'frame_paths': frame_files
                                    })

        print(f"[{self.split.upper()} DATASET] Discovered {len(self.samples)} total samples:")
        images_count = sum(1 for s in self.samples if s['type'] == 'image')
        videos_count = sum(1 for s in self.samples if s['type'] == 'video')
        csam_count = sum(1 for s in self.samples if s['binary_label'] == 1)
        non_csam_count = sum(1 for s in self.samples if s['binary_label'] == 0)
        print(f"  - Images: {images_count} | Videos: {videos_count}")
        print(f"  - CSAM Samples: {csam_count} (CSAM_sens + CSAM_non_sens) | Non-CSAM Samples: {non_csam_count}")

    def __len__(self) -> int:
        return len(self.samples)

    def sample_video_frames(self, frame_paths: List[str]) -> List[Image.Image]:
        """Uniformly samples max_video_frames from the video frame list."""
        n = len(frame_paths)
        if n <= self.max_video_frames:
            sampled_paths = frame_paths
        else:
            indices = [int(i * (n - 1) / (self.max_video_frames - 1)) for i in range(self.max_video_frames)]
            sampled_paths = [frame_paths[i] for i in indices]

        images = []
        for p in sampled_paths:
            try:
                img = Image.open(p).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"Warning: Failed to load frame {p}: {e}")
        return images

    def get_sampled_frame_paths(self, frame_paths: List[str]) -> List[str]:
        """Returns the list of uniformly sampled file paths."""
        n = len(frame_paths)
        if n <= self.max_video_frames:
            return frame_paths
        indices = [int(i * (n - 1) / (self.max_video_frames - 1)) for i in range(self.max_video_frames)]
        return [frame_paths[i] for i in indices]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        if item['type'] == 'image':
            try:
                img = Image.open(item['path']).convert('RGB')
                if self.transform:
                    tensor_data = self.transform(img)
                else:
                    tensor_data = img
            except Exception as e:
                print(f"Warning: Error opening image {item['path']}: {e}")
                tensor_data = None
                img = None

            return {
                **item,
                'raw_image': img,
                'tensor_data': tensor_data,
                'is_video': False
            }
        else:
            # Video item: sample frames
            frames = self.sample_video_frames(item['frame_paths'])
            if self.transform and len(frames) > 0:
                frame_tensors = [self.transform(f) for f in frames]
                tensor_data = torch.stack(frame_tensors) if len(frame_tensors) > 0 else None
            else:
                tensor_data = frames

            return {
                **item,
                'raw_frames': frames,
                'tensor_data': tensor_data,
                'is_video': True
            }

    def get_dataframe(self) -> pd.DataFrame:
        """Returns metadata summary dataframe."""
        return pd.DataFrame([{
            'id': s['id'],
            'path': s['path'],
            'type': s['type'],
            'class_name': s['class_name'],
            'binary_label': s['binary_label'],
            'label_3class': s['label_3class'],
            'split': s['split'],
            'num_frames': s['num_frames']
        } for s in self.samples])


def collate_multimodal_fn(batch):
    """Custom collator for variable-length frame sequences and images."""
    return batch
