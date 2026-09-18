import os
import sys
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataset.dataset import CSAMDataset


import json

LOGIC_CACHE_FILE = os.path.join(os.path.dirname(__file__), 'logic_vector_cache.json')


def load_logic_cache():
    if os.path.exists(LOGIC_CACHE_FILE):
        try:
            with open(LOGIC_CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_logic_cache(cache):
    try:
        with open(LOGIC_CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"[WARNING] Could not save logic cache: {e}")


def build_caption_map(image_dir="dataset_updated_organized"):
    """
    Builds a global caption lookup dictionary across all train/val/test caption files.
    """
    caption_files = [
        os.path.join(image_dir, "csam_non_sens.csv"),
        os.path.join(image_dir, "non_csam.csv"),
        os.path.join(image_dir, "train", "train_csam_non_sens.csv"),
        os.path.join(image_dir, "train", "train_non_csam.csv"),
        os.path.join(image_dir, "val", "val_csam_non_sens.csv"),
        os.path.join(image_dir, "val", "val_non_csam.csv"),
        os.path.join(image_dir, "test", "test_csam_non_sens.csv"),
        os.path.join(image_dir, "test", "test_non_csam.csv")
    ]
    
    caption_map = {}
    for cap_file in caption_files:
        if os.path.exists(cap_file):
            df = pd.read_csv(cap_file)
            for _, row in df.iterrows():
                fn = str(row['filename']).strip() if 'filename' in row else str(row['image name']).strip()
                txt = ""
                if 'text_description' in row and pd.notna(row['text_description']):
                    txt = str(row['text_description']).strip()
                elif 'text' in row and pd.notna(row['text']):
                    txt = str(row['text']).strip()
                caption_map[fn] = txt
    return caption_map


def prepare_client_csv(csv_path, caption_map):
    """
    Ensures that the client CSV has a 'text' column.
    Saves and returns the path to prepared CSV file.
    """
    df = pd.read_csv(csv_path)
    prepared_csv_path = csv_path.replace('.csv', '_prepared.csv')
    
    if 'text' in df.columns and not df['text'].isnull().any():
        return csv_path if os.path.exists(csv_path) else prepared_csv_path
        
    def get_text(row):
        if 'text' in row and pd.notna(row['text']) and str(row['text']).strip():
            return str(row['text']).strip()
        img_name = str(row['image name']).strip() if 'image name' in row else str(row['image']).strip()
        return caption_map.get(img_name, "")
        
    df['text'] = df.apply(get_text, axis=1)
    df.to_csv(prepared_csv_path, index=False)
    return prepared_csv_path


def load_real_client_dataloaders(splits_dir="federated_splits", image_dir="dataset_updated_organized", batch_size=4, num_clients=5):
    """
    Loads PyTorch DataLoaders for each client (train, val, test splits) from federated_splits.
    """
    caption_map = build_caption_map(image_dir=image_dir)
    client_dataloaders = {}

    for i in range(1, num_clients + 1):
        client_name = f"client_{i}"
        client_dir = os.path.join(splits_dir, client_name)
        
        train_csv = os.path.join(client_dir, "train.csv")
        val_csv = os.path.join(client_dir, "val.csv")
        test_csv = os.path.join(client_dir, "test.csv")
        
        if not (os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv)):
            raise FileNotFoundError(f"Missing CSV files for {client_name} in {client_dir}")
            
        prep_train_csv = prepare_client_csv(train_csv, caption_map)
        prep_val_csv = prepare_client_csv(val_csv, caption_map)
        prep_test_csv = prepare_client_csv(test_csv, caption_map)

        train_ds = CSAMDataset(csv_file=prep_train_csv, image_dir=image_dir, precompute_logic=False, skip_caption_loading=True)
        val_ds = CSAMDataset(csv_file=prep_val_csv, image_dir=image_dir, precompute_logic=False, skip_caption_loading=True)
        test_ds = CSAMDataset(csv_file=prep_test_csv, image_dir=image_dir, precompute_logic=False, skip_caption_loading=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        client_dataloaders[client_name] = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'train_ds': train_ds,
            'val_ds': val_ds,
            'test_ds': test_ds
        }
        
    return client_dataloaders


class CSAMFedDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.images = torch.rand(num_samples, 3, 384, 384)
        
        safe_texts = [
            'children playing in park', 'family vacation photo', 'school group picture',
            'kids soccer match', 'birthday party celebration', 'playground fun time',
            'classroom learning activity', 'pet dog with family', 'beach holiday photo',
            'nature hiking trip'
        ]
        concerning_texts = [
            'child looks scared and alone', 'kid crying in dark room', 'forced to keep secret',
            'threatening message to minor', 'inappropriate content with child', 'scared child with adult',
            'bullying at school', 'child in distress', 'coercion and threats', 'abuse victim testimony'
        ]
        
        all_texts = safe_texts + concerning_texts
        self.texts = [random.choice(all_texts) for _ in range(num_samples)]
        self.logic_vectors = torch.rand(num_samples, 5)
        self.labels = torch.tensor([1 if random.random() < 0.3 else 0 for _ in range(num_samples)], dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'text': self.texts[idx],
            'logic_vector': self.logic_vectors[idx],
            'label': self.labels[idx]
        }


def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    logic_vectors = torch.stack([item['logic_vector'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    return {
        'images': images,
        'texts': texts,
        'logic_vectors': logic_vectors,
        'labels': labels
    }


def create_client_dataloaders(num_clients, samples_per_client, batch_size, iid=True):
    total_samples = num_clients * samples_per_client
    full_dataset = CSAMFedDataset(total_samples)
    dataloaders = []
    
    if iid:
        indices = list(range(total_samples))
        random.shuffle(indices)
        for i in range(num_clients):
            client_indices = indices[i * samples_per_client : (i + 1) * samples_per_client]
            client_dataset = torch.utils.data.Subset(full_dataset, client_indices)
            dataloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
            dataloaders.append(dataloader)
    else:
        indices_by_label = {0: [], 1: []}
        for idx in range(total_samples):
            label = full_dataset.labels[idx].item()
            indices_by_label[label].append(idx)
        all_indices = indices_by_label[0] + indices_by_label[1]
        for i in range(num_clients):
            client_indices = all_indices[i * samples_per_client : (i + 1) * samples_per_client]
            client_dataset = torch.utils.data.Subset(full_dataset, client_indices)
            dataloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
            dataloaders.append(dataloader)
            
    return dataloaders

