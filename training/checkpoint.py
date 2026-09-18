import os
import torch

def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """
    Saves state dictionary of model and optimizer, epoch, and validation metrics.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics
    }
    torch.save(checkpoint, save_path)
    print(f"[OK] Saved checkpoint to {save_path}")

def load_checkpoint(model, optimizer=None, checkpoint_path=None, device="cpu"):
    """
    Loads checkpoint into model and optimizer.
    """
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"Checkpoint file {checkpoint_path} not found. Starting from scratch.")
        return 0, {}
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})
    print(f"[OK] Loaded checkpoint from {checkpoint_path} (Epoch {epoch})")
    return epoch, metrics
