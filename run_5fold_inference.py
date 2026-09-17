"""
5-Fold Inference & Qualitative Analysis Script for Fine-Tuned Full CSAM Model
=============================================================================
Evaluates the fine-tuned full CSAM model pipeline (SigLIP + MPNet + Cross-Attention + 
Logic Encoder + Fusion Transformer + Classifier) across all 5 folds of the 5-fold CV split.

Generates:
1. Per-fold detailed prediction CSV files with sample-level classifications and probabilities.
2. Per-fold qualitative analysis split files:
   - Correctly classified CSAM samples (TP)
   - Correctly classified Non-CSAM samples (TN)
   - All misclassified samples (FP & FN)
3. Confusion matrices and detailed JSON metrics for every fold.
4. Aggregated 5-fold cross-validation results and summary tables (emphasizing Fold 1).
"""

import os
import sys
import json
import gc
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
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CV_CHECKPOINTS_DIR = os.path.join(BASE_DIR, "cv_checkpoints")
IMAGE_DIR = os.path.join(BASE_DIR, "dataset_updated_organized")
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "results", "5fold_inference")
BATCH_SIZE = 4
CSAM_WEIGHT = 2.0
N_FOLDS = 5


def run_5fold_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("5-FOLD FULL CSAM MODEL INFERENCE & QUALITATIVE ANALYSIS")
    print("=" * 70)
    print(f"Device        : {device}")
    if device.type == "cuda":
        print(f"GPU Name      : {torch.cuda.get_device_name(0)}")
    print(f"Checkpoints   : {CV_CHECKPOINTS_DIR}")
    print(f"Image Dir     : {IMAGE_DIR}")
    print(f"Output Dir    : {OUTPUT_BASE_DIR}")
    print(f"Folds         : {N_FOLDS}")
    print("=" * 70)
    
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    all_fold_summary_metrics = []
    all_predictions_df_list = []
    
    for fold in range(1, N_FOLDS + 1):
        print("\n\n" + "=" * 70)
        print(f"RUNNING INFERENCE - FOLD {fold}/{N_FOLDS}")
        print("=" * 70)
        
        # Determine checkpoint path
        ckpt_path = os.path.join(CV_CHECKPOINTS_DIR, f"fold_{fold}", "best_model.pt")
        if not os.path.exists(ckpt_path):
            # Fallback to ablation fold path if main fold directory absent
            ckpt_path = os.path.join(CV_CHECKPOINTS_DIR, "ablation_A0_full_model", f"fold_{fold}", "best_model.pt")
            
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[ERROR] Checkpoint not found for Fold {fold}: {ckpt_path}")
            
        print(f"[OK] Found model checkpoint: {ckpt_path}")
        
        # Determine prepared test CSV path
        prepared_test_csv = os.path.join(CV_CHECKPOINTS_DIR, f"fold_{fold}", "prepared_data", "test.csv")
        if not os.path.exists(prepared_test_csv):
            prepared_test_csv = os.path.join(BASE_DIR, "5_fold_splits", f"fold_{fold}", "test.csv")
            
        if not os.path.exists(prepared_test_csv):
            raise FileNotFoundError(f"[ERROR] Test CSV not found for Fold {fold}: {prepared_test_csv}")
            
        print(f"[OK] Found prepared test CSV: {prepared_test_csv}")
        
        fold_output_dir = os.path.join(OUTPUT_BASE_DIR, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # Load dataset
        print(f"\n[1/4] Loading Fold {fold} Test Dataset...")
        test_dataset = CSAMDataset(
            csv_file=prepared_test_csv,
            image_dir=IMAGE_DIR,
            precompute_logic=True,
            skip_caption_loading=True
        )
        print(f"[OK] Total test samples in Fold {fold}: {len(test_dataset)}")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        # Load model
        print(f"\n[2/4] Loading Full CSAM Pipeline Model for Fold {fold}...")
        model = FullCSAMPipeline(mode="full").to(device)
        load_checkpoint(model, checkpoint_path=ckpt_path, device=device)
        model.eval()
        
        criterion = CSAMLoss(csam_weight=CSAM_WEIGHT)
        
        # Run inference
        print(f"\n[3/4] Executing Inference Loop for Fold {fold}...")
        all_targets = []
        all_preds = []
        all_probs = []
        all_image_names = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Fold {fold} Progress"):
                images = batch["image"].to(device)
                texts = batch["text"]
                logic_vectors = batch["logic_vector"].to(device)
                labels = batch["label"].to(device)
                image_names = batch["image_name"]
                
                logits, attention = model(images, texts, logic_vectors)
                loss = criterion(logits, labels)
                total_loss += loss.item() * len(labels)
                
                probs = torch.softmax(logits, dim=-1)[:, 1]
                preds = torch.argmax(logits, dim=-1)
                
                all_targets.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())
                all_image_names.extend(image_names)
                
        avg_loss = total_loss / len(test_dataset)
        
        # Calculate metrics
        print(f"\n[4/4] Computing Fold {fold} Metrics & Structuring Qualitative Splits...")
        metrics = compute_metrics(
            y_true=all_targets,
            y_pred=all_preds,
            y_prob=all_probs,
            mode="binary"
        )
        metrics["Fold"] = fold
        metrics["Test_Loss"] = avg_loss
        
        # Create detailed prediction DataFrame
        df_meta = test_dataset.data.copy()
        
        # Align prediction fields
        pred_records = []
        for idx in range(len(all_targets)):
            gt = all_targets[idx]
            pred = all_preds[idx]
            prob = all_probs[idx]
            img_name = all_image_names[idx]
            
            # Classification status designation
            if gt == 1 and pred == 1:
                status = "Correct_CSAM_TP"
            elif gt == 0 and pred == 0:
                status = "Correct_Non_CSAM_TN"
            elif gt == 0 and pred == 1:
                status = "Misclassified_Non_CSAM_FP"
            elif gt == 1 and pred == 0:
                status = "Misclassified_CSAM_FN"
            else:
                status = "Unknown"
                
            # Retrieve metadata fields if available
            row_meta = df_meta.iloc[idx] if idx < len(df_meta) else {}
            filepath = row_meta.get("filepath", row_meta.get("image_path", ""))
            subtype = row_meta.get("subtype", "")
            group_id = row_meta.get("group_id", "")
            caption = row_meta.get("text", "")
            
            pred_records.append({
                "fold": fold,
                "sample_index": idx,
                "image_name": img_name,
                "filepath": filepath,
                "subtype": subtype,
                "group_id": group_id,
                "ground_truth_label": gt,
                "ground_truth_str": "CSAM" if gt == 1 else "NON_CSAM",
                "predicted_label": pred,
                "predicted_str": "CSAM" if pred == 1 else "NON_CSAM",
                "csam_probability": round(prob, 6),
                "classification_status": status,
                "caption": caption
            })
            
        fold_pred_df = pd.DataFrame(pred_records)
        all_predictions_df_list.append(fold_pred_df)
        
        # Qualitative Splits
        correct_csam_df = fold_pred_df[fold_pred_df["classification_status"] == "Correct_CSAM_TP"].copy()
        correct_non_csam_df = fold_pred_df[fold_pred_df["classification_status"] == "Correct_Non_CSAM_TN"].copy()
        misclassified_df = fold_pred_df[fold_pred_df["classification_status"].isin(["Misclassified_Non_CSAM_FP", "Misclassified_CSAM_FN"])].copy()
        
        # Save per-fold files
        pred_csv_path = os.path.join(fold_output_dir, f"fold_{fold}_predictions.csv")
        fold_pred_df.to_csv(pred_csv_path, index=False)
        
        correct_csam_path = os.path.join(fold_output_dir, f"fold_{fold}_correct_csam.csv")
        correct_csam_df.to_csv(correct_csam_path, index=False)
        
        correct_non_csam_path = os.path.join(fold_output_dir, f"fold_{fold}_correct_non_csam.csv")
        correct_non_csam_df.to_csv(correct_non_csam_path, index=False)
        
        misclassified_path = os.path.join(fold_output_dir, f"fold_{fold}_all_misclassified.csv")
        misclassified_df.to_csv(misclassified_path, index=False)
        
        metrics_json_path = os.path.join(fold_output_dir, f"fold_{fold}_metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=4)
            
        cm_png_path = os.path.join(fold_output_dir, f"fold_{fold}_confusion_matrix.png")
        plot_and_save_confusion_matrix(
            y_true=all_targets,
            y_pred=all_preds,
            class_names=["Non_CSAM", "CSAM"],
            save_path=cm_png_path,
            title=f"Full CSAM Model - Fold {fold} Outer Test Confusion Matrix"
        )
        
        all_fold_summary_metrics.append(metrics)
        
        print(f"[SAVED] Fold {fold} Predictions CSV     -> {pred_csv_path}")
        print(f"[SAVED] Fold {fold} Correct CSAM CSV   -> {correct_csam_path} (Count: {len(correct_csam_df)})")
        print(f"[SAVED] Fold {fold} Correct NonCSAM CSV-> {correct_non_csam_path} (Count: {len(correct_non_csam_df)})")
        print(f"[SAVED] Fold {fold} Misclassified CSV  -> {misclassified_path} (Count: {len(misclassified_df)})")
        print(f"[SAVED] Fold {fold} Metrics JSON       -> {metrics_json_path}")
        print(f"[SAVED] Fold {fold} Confusion Matrix   -> {cm_png_path}")
        print("-" * 70)
        print(f"FOLD {fold} SUMMARY:")
        print(f"  Accuracy: {metrics['Accuracy']:.4f} | Precision: {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f} | F1: {metrics['F1_Score']:.4f}")
        print(f"  TP: {metrics['TP']} | TN: {metrics['TN']} | FP: {metrics['FP']} | FN: {metrics['FN']}")
        print("-" * 70)
        
        # Cleanup GPU memory between folds
        del model, test_dataset, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # ----------------------------------------------------
    # Consolidated 5-Fold Export
    # ----------------------------------------------------
    print("\n\n" + "=" * 70)
    print("CONSOLIDATING 5-FOLD CV INFERENCE & QUALITATIVE RESULTS")
    print("=" * 70)
    
    full_5fold_preds_df = pd.concat(all_predictions_df_list, ignore_index=True)
    all_preds_csv_path = os.path.join(OUTPUT_BASE_DIR, "5fold_all_predictions.csv")
    full_5fold_preds_df.to_csv(all_preds_csv_path, index=False)
    print(f"[SAVED] Combined 5-Fold Predictions CSV -> {all_preds_csv_path} (Total Rows: {len(full_5fold_preds_df)})")
    
    summary_metrics_df = pd.DataFrame(all_fold_summary_metrics)
    summary_csv_path = os.path.join(OUTPUT_BASE_DIR, "5fold_summary_metrics.csv")
    summary_metrics_df.to_csv(summary_csv_path, index=False)
    
    summary_json_path = os.path.join(OUTPUT_BASE_DIR, "5fold_summary_metrics.json")
    with open(summary_json_path, "w") as f:
        json.dump(all_fold_summary_metrics, f, indent=4)
    print(f"[SAVED] 5-Fold Summary Metrics CSV      -> {summary_csv_path}")
    print(f"[SAVED] 5-Fold Summary Metrics JSON     -> {summary_json_path}")
    
    # 5-Fold Mean & Std
    accs = [m["Accuracy"] for m in all_fold_summary_metrics]
    precs = [m["Precision"] for m in all_fold_summary_metrics]
    recs = [m["Recall"] for m in all_fold_summary_metrics]
    f1s = [m["F1_Score"] for m in all_fold_summary_metrics]
    
    print("\n" + "=" * 70)
    print("FINAL 5-FOLD CROSS-VALIDATION STATISTICAL SUMMARY")
    print("=" * 70)
    print(f"Accuracy  : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Precision : {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"Recall    : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    print(f"F1-Score  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print("=" * 70)
    print("[SUCCESS] 5-Fold Inference & Qualitative Analysis execution ready!")


if __name__ == "__main__":
    run_5fold_inference()
