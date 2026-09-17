"""
Evaluation & Metrics Utilities for Baseline Tests
=================================================
Calculates:
- Overall collective metrics (Accuracy, Precision, Recall, Specificity, F1, ROC-AUC)
- Confusion Matrix & Classification Report
- Per-modality breakdown (Collective Overall, Images subset, Videos subset)
- Confusion Matrix Plotting and CSV export
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[List[float]] = None,
    mode: str = 'binary'
) -> Dict[str, Any]:
    """
    Computes standard classification metrics.
    """
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    acc = accuracy_score(y_true_np, y_pred_np)
    
    if mode == 'binary':
        prec = precision_score(y_true_np, y_pred_np, zero_division=0)
        rec = recall_score(y_true_np, y_pred_np, zero_division=0)
        f1 = f1_score(y_true_np, y_pred_np, zero_division=0)
        
        cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        auc = 0.0
        if y_prob is not None and len(np.unique(y_true_np)) > 1:
            try:
                auc = roc_auc_score(y_true_np, y_prob)
            except Exception:
                auc = 0.0
                
        return {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'Specificity': specificity,
            'F1_Score': f1,
            'ROC_AUC': auc,
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'Total_Samples': len(y_true_np)
        }
    else:
        # Multi-class
        prec_macro = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        rec_macro = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        f1_macro = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
        
        return {
            'Accuracy': acc,
            'Precision_Macro': prec_macro,
            'Recall_Macro': rec_macro,
            'F1_Macro': f1_macro,
            'F1_Weighted': f1_weighted,
            'Total_Samples': len(y_true_np)
        }


def plot_and_save_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: str,
    title: str = "Baseline Confusion Matrix"
):
    """Plots and saves normalized and count confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(7, 6))
    annot = np.array([[f"{count}\n({norm:.1%})" for count, norm in zip(row_c, row_n)] 
                      for row_c, row_n in zip(cm, cm_norm)])
    
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 12, "weight": "bold"}
    )
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('Ground Truth Label', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix saved to {save_path}")


def generate_collective_summary(
    results_df: pd.DataFrame,
    model_name: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Generates collective overall report, modality breakdown, and exports results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed predictions CSV
    csv_file = output_path / f"{model_name.lower()}_detailed_predictions.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"Detailed predictions saved to {csv_file}")

    # 1. Overall Collective Metrics
    overall_metrics = compute_metrics(
        results_df['ground_truth'].tolist(),
        results_df['prediction'].tolist(),
        results_df['confidence'].tolist() if 'confidence' in results_df else None,
        mode='binary'
    )

    # 2. Image subset metrics
    img_df = results_df[results_df['type'] == 'image']
    img_metrics = compute_metrics(
        img_df['ground_truth'].tolist(),
        img_df['prediction'].tolist(),
        img_df['confidence'].tolist() if 'confidence' in img_df else None,
        mode='binary'
    ) if len(img_df) > 0 else {}

    # 3. Video subset metrics
    vid_df = results_df[results_df['type'] == 'video']
    vid_metrics = compute_metrics(
        vid_df['ground_truth'].tolist(),
        vid_df['prediction'].tolist(),
        vid_df['confidence'].tolist() if 'confidence' in vid_df else None,
        mode='binary'
    ) if len(vid_df) > 0 else {}

    # Plot Confusion Matrix
    cm_path = str(output_path / f"{model_name.lower()}_confusion_matrix.png")
    plot_and_save_confusion_matrix(
        results_df['ground_truth'].tolist(),
        results_df['prediction'].tolist(),
        class_names=['Non-CSAM (0)', 'CSAM (1)'],
        save_path=cm_path,
        title=f"{model_name} Baseline Test - Confusion Matrix"
    )

    # Build Summary Table
    summary_data = [
        {
            'Evaluation Scope': 'Collective Overall (Images + Videos)',
            'Total Samples': overall_metrics.get('Total_Samples', 0),
            'Accuracy': f"{overall_metrics.get('Accuracy', 0):.4f}",
            'Precision': f"{overall_metrics.get('Precision', 0):.4f}",
            'Recall (Sensitivity)': f"{overall_metrics.get('Recall', 0):.4f}",
            'Specificity': f"{overall_metrics.get('Specificity', 0):.4f}",
            'F1-Score': f"{overall_metrics.get('F1_Score', 0):.4f}",
            'ROC-AUC': f"{overall_metrics.get('ROC_AUC', 0):.4f}",
            'TP': overall_metrics.get('TP', 0),
            'TN': overall_metrics.get('TN', 0),
            'FP': overall_metrics.get('FP', 0),
            'FN': overall_metrics.get('FN', 0)
        }
    ]

    if img_metrics:
        summary_data.append({
            'Evaluation Scope': 'Images Modality Subset',
            'Total Samples': img_metrics.get('Total_Samples', 0),
            'Accuracy': f"{img_metrics.get('Accuracy', 0):.4f}",
            'Precision': f"{img_metrics.get('Precision', 0):.4f}",
            'Recall (Sensitivity)': f"{img_metrics.get('Recall', 0):.4f}",
            'Specificity': f"{img_metrics.get('Specificity', 0):.4f}",
            'F1-Score': f"{img_metrics.get('F1_Score', 0):.4f}",
            'ROC-AUC': f"{img_metrics.get('ROC_AUC', 0):.4f}",
            'TP': img_metrics.get('TP', 0),
            'TN': img_metrics.get('TN', 0),
            'FP': img_metrics.get('FP', 0),
            'FN': img_metrics.get('FN', 0)
        })

    if vid_metrics:
        summary_data.append({
            'Evaluation Scope': 'Videos Modality Subset',
            'Total Samples': vid_metrics.get('Total_Samples', 0),
            'Accuracy': f"{vid_metrics.get('Accuracy', 0):.4f}",
            'Precision': f"{vid_metrics.get('Precision', 0):.4f}",
            'Recall (Sensitivity)': f"{vid_metrics.get('Recall', 0):.4f}",
            'Specificity': f"{vid_metrics.get('Specificity', 0):.4f}",
            'F1-Score': f"{vid_metrics.get('F1_Score', 0):.4f}",
            'ROC-AUC': f"{vid_metrics.get('ROC_AUC', 0):.4f}",
            'TP': vid_metrics.get('TP', 0),
            'TN': vid_metrics.get('TN', 0),
            'FP': vid_metrics.get('FP', 0),
            'FN': vid_metrics.get('FN', 0)
        })

    summary_df = pd.DataFrame(summary_data)
    summary_csv = output_path / f"{model_name.lower()}_baseline_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "="*80)
    print(f"📊 {model_name} BASELINE TEST - OVERALL COLLECTIVE RESULTS")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)

    return {
        'overall': overall_metrics,
        'image': img_metrics,
        'video': vid_metrics,
        'summary_df': summary_df,
        'predictions_csv': str(csv_file),
        'confusion_matrix_png': cm_path
    }
