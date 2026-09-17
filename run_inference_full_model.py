"""
Inference Script for Fine-Tuned Full CSAM Model
================================================
Evaluates the fine-tuned full CSAM model (SigLIP + MPNet + Cross-Attention + Logic Encoder + Fusion Transformer + Classifier)
on the test split of dataset_updated_organized.

Run directly via:
    python run_inference_full_model.py
"""

import os
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

from dataset.dataset import CSAMDataset
from models.full_pipeline import FullCSAMPipeline
from training.checkpoint import load_checkpoint
from training.loss import CSAMLoss
from metrics_utils import compute_metrics, plot_and_save_confusion_matrix


# ============================================================
# FIXED CONFIGURATION VARIABLES
# ============================================================
CHECKPOINT_PATH = "checkpoints/complete_model.pt"
TEST_CSV = "dataset_updated_organized/test/metadata.csv" if os.path.exists("dataset_updated_organized/test/metadata.csv") else ("5_fold_splits/fold_1/test.csv" if os.path.exists("5_fold_splits/fold_1/test.csv") else "dataset_updated_organized/metadata.csv")
IMAGE_DIR = "dataset_updated_organized"
BATCH_SIZE = 4
OUTPUT_DIR = "results/full_model_inference"
CSAM_WEIGHT = 2.0


def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("FULL CSAM MODEL - TEST INFERENCE")
    print("=" * 70)
    print(f"Device      : {device}")
    if device.type == "cuda":
        print(f"GPU Name    : {torch.cuda.get_device_name(0)}")
    print(f"Test CSV    : {TEST_CSV}")
    print(f"Image Dir   : {IMAGE_DIR}")
    print(f"Checkpoint  : {CHECKPOINT_PATH}")
    print(f"Batch Size  : {BATCH_SIZE}")
    print(f"Output Dir  : {OUTPUT_DIR}")
    print("=" * 70)
    
    # ----------------------------------------------------
    # Checkpoint Path Fallback Check
    # ----------------------------------------------------
    ckpt_path = CHECKPOINT_PATH
    if not os.path.exists(ckpt_path):
        candidate_paths = [
            "checkpoints/best_model.pt",
            "cv_checkpoints/ablation_A0_full_model/fold_1/best_model.pt"
        ]
        found = False
        for candidate in candidate_paths:
            if os.path.exists(candidate):
                print(f"[WARNING] Checkpoint '{ckpt_path}' not found.")
                print(f"[INFO] Using fallback checkpoint: '{candidate}'")
                ckpt_path = candidate
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")
            
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ----------------------------------------------------
    # Load Test Dataset & DataLoader
    # ----------------------------------------------------
    print("\n[1/4] Loading Test Dataset...")
    test_dataset = CSAMDataset(
        csv_file=TEST_CSV,
        image_dir=IMAGE_DIR,
        precompute_logic=True
    )
    print(f"[OK] Total test samples loaded: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    print(f"[OK] DataLoader initialized ({len(test_loader)} batches)")
    
    # ----------------------------------------------------
    # Load Fine-Tuned Model Pipeline
    # ----------------------------------------------------
    print("\n[2/4] Initializing Full CSAM Pipeline & Loading Checkpoint...")
    model = FullCSAMPipeline(mode="full").to(device)
    
    load_checkpoint(model, checkpoint_path=ckpt_path, device=device)
    model.eval()
    print(f"[OK] Fine-tuned weights loaded from: {ckpt_path}")
    
    criterion = CSAMLoss(csam_weight=CSAM_WEIGHT)
    
    # ----------------------------------------------------
    # Inference Loop
    # ----------------------------------------------------
    print("\n[3/4] Running Inference on Test Set...")
    all_targets = []
    all_preds = []
    all_probs = []
    all_image_names = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference Progress"):
            images = batch["image"].to(device)
            texts = batch["text"]
            logic_vectors = batch["logic_vector"].to(device)
            labels = batch["label"].to(device)
            image_names = batch["image_name"]
            
            # Forward Pass
            logits, attention = model(images, texts, logic_vectors)
            
            # Loss Calculation
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            
            # Probabilities & Predictions
            probs = torch.softmax(logits, dim=-1)[:, 1]  # Probability of CSAM class (1)
            preds = torch.argmax(logits, dim=-1)
            
            all_targets.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_image_names.extend(image_names)
            
    avg_loss = total_loss / len(test_dataset)
    
    # ----------------------------------------------------
    # Compute Metrics
    # ----------------------------------------------------
    print("\n[4/4] Computing Metrics and Saving Output Artifacts...")
    metrics = compute_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        y_prob=all_probs,
        mode="binary"
    )
    metrics["Test_Loss"] = avg_loss
    
    # Print Results Summary
    print("\n" + "=" * 70)
    print("INFERENCE EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Samples   : {metrics['Total_Samples']}")
    print(f"Test Loss       : {metrics['Test_Loss']:.4f}")
    print(f"Accuracy        : {metrics['Accuracy']:.4f} ({metrics['Accuracy'] * 100:.2f}%)")
    print(f"Precision       : {metrics['Precision']:.4f}")
    print(f"Recall (Sens.)  : {metrics['Recall']:.4f}")
    print(f"Specificity     : {metrics['Specificity']:.4f}")
    print(f"F1-Score        : {metrics['F1_Score']:.4f}")
    print(f"ROC-AUC         : {metrics['ROC_AUC']:.4f}")
    print("-" * 70)
    print("Confusion Matrix:")
    print(f"  True Positives  (TP) : {metrics['TP']}")
    print(f"  True Negatives  (TN) : {metrics['TN']}")
    print(f"  False Positives (FP) : {metrics['FP']}")
    print(f"  False Negatives (FN) : {metrics['FN']}")
    print("=" * 70)
    
    # Save Detailed Predictions CSV
    pred_df = pd.DataFrame({
        "image_name": all_image_names,
        "ground_truth_label": all_targets,
        "predicted_label": all_preds,
        "csam_probability": all_probs
    })
    pred_csv_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"[SAVED] Detailed predictions -> {pred_csv_path}")
    
    # Save Metrics JSON
    json_metrics_path = os.path.join(OUTPUT_DIR, "inference_metrics.json")
    with open(json_metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[SAVED] Metrics JSON         -> {json_metrics_path}")
    
    # Save Confusion Matrix Plot
    cm_plot_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plot_and_save_confusion_matrix(
        y_true=all_targets,
        y_pred=all_preds,
        class_names=["Non_CSAM", "CSAM"],
        save_path=cm_plot_path,
        title="Fine-Tuned Full CSAM Model - Test Confusion Matrix"
    )
    print(f"[SAVED] Confusion Matrix Plot-> {cm_plot_path}")
    print("=" * 70)
    print("[SUCCESS] Inference pipeline execution completed!")


if __name__ == "__main__":
    run_inference()
