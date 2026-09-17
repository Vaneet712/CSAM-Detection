import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
)

def compute_detailed_metrics(all_labels, all_preds, all_scores=None):
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    total = len(all_labels)
    acc = (tp + tn) / total * 100 if total > 0 else 0.0
    sens = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    spec = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0
    fpr = (fp / (tn + fp) * 100) if (tn + fp) > 0 else 0.0
    fnr = (fn / (tp + fn) * 100) if (tp + fn) > 0 else 0.0

    b_prec = precision_score(all_labels, all_preds, average='binary', zero_division=0) * 100
    b_rec = recall_score(all_labels, all_preds, average='binary', zero_division=0) * 100
    b_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0) * 100

    m_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    m_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    m_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100

    w_prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    w_rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    w_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100

    prec_cls = precision_score(all_labels, all_preds, average=None, zero_division=0) * 100
    rec_cls = recall_score(all_labels, all_preds, average=None, zero_division=0) * 100
    f1_cls = f1_score(all_labels, all_preds, average=None, zero_division=0) * 100

    roc_auc = 0.0
    if all_scores is not None and len(np.unique(all_labels)) > 1:
        roc_auc = roc_auc_score(all_labels, all_scores) * 100

    return {
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
        'Accuracy': acc, 'Sensitivity': sens, 'Specificity': spec,
        'FPR': fpr, 'FNR': fnr, 'Bin_Prec': b_prec, 'Bin_Rec': b_rec, 'Bin_F1': b_f1,
        'Mac_Prec': m_prec, 'Mac_Rec': m_rec, 'Mac_F1': m_f1,
        'Wt_Prec': w_prec, 'Wt_Rec': w_rec, 'Wt_F1': w_f1,
        'C0_Prec': prec_cls[0] if len(prec_cls) > 0 else 0.0,
        'C0_Rec': rec_cls[0] if len(rec_cls) > 0 else 0.0,
        'C0_F1': f1_cls[0] if len(f1_cls) > 0 else 0.0,
        'C1_Prec': prec_cls[1] if len(prec_cls) > 1 else 0.0,
        'C1_Rec': rec_cls[1] if len(rec_cls) > 1 else 0.0,
        'C1_F1': f1_cls[1] if len(f1_cls) > 1 else 0.0,
        'ROC_AUC': roc_auc,
        'CM': cm
    }

def print_fold_report(fold_name, m):
    print("\n" + "=" * 65)
    print(f"      DETAILED EVALUATION REPORT: FOLD {fold_name}")
    print("=" * 65)
    print(f"  TP: {m['TP']:<5} | FP: {m['FP']:<5}")
    print(f"  TN: {m['TN']:<5} | FN: {m['FN']:<5}")
    print(f"\n  Accuracy      : {m['Accuracy']:.2f}%")
    print(f"  ROC AUC       : {m['ROC_AUC']:.2f}%")
    print(f"  Sensitivity   : {m['Sensitivity']:.2f}% (Detection rate)")
    print(f"  Specificity   : {m['Specificity']:.2f}% (Safe retention)")
    print(f"  FPR           : {m['FPR']:.2f}%")
    print(f"  Miss Rate(FNR): {m['FNR']:.2f}%")
    print("\n  --- Class-Specific ---")
    print(f"  Class 0 (Safe)-> Prec: {m['C0_Prec']:.2f}% | Rec: {m['C0_Rec']:.2f}% | F1: {m['C0_F1']:.2f}%")
    print(f"  Class 1 (CSAM)-> Prec: {m['C1_Prec']:.2f}% | Rec: {m['C1_Rec']:.2f}% | F1: {m['C1_F1']:.2f}%")
    print("\n  --- Aggregated ---")
    print(f"  Binary   -> Prec: {m['Bin_Prec']:.2f}% | Rec: {m['Bin_Rec']:.2f}% | F1: {m['Bin_F1']:.2f}%")
    print(f"  Macro    -> Prec: {m['Mac_Prec']:.2f}% | Rec: {m['Mac_Rec']:.2f}% | F1: {m['Mac_F1']:.2f}%")
    print(f"  Weighted -> Prec: {m['Wt_Prec']:.2f}% | Rec: {m['Wt_Rec']:.2f}% | F1: {m['Wt_F1']:.2f}%")
    print("=" * 65)

def main():
    predictions_file = 'results/5fold_inference/5fold_all_predictions.csv'
    if not os.path.exists(predictions_file):
        print(f"Error: Could not find {predictions_file}")
        return

    print("=" * 65)
    print("      MAIN MODEL 5-FOLD CV COMPREHENSIVE EVALUATION")
    print("=" * 65)

    df = pd.read_csv(predictions_file)
    all_fold_metrics = []

    for fold in sorted(df['fold'].unique()):
        sub_df = df[df['fold'] == fold]
        y_true = sub_df['ground_truth_label'].values
        y_score = sub_df['csam_probability'].values
        y_pred = sub_df['predicted_label'].values if 'predicted_label' in sub_df.columns else (y_score >= 0.5).astype(int)

        m = compute_detailed_metrics(y_true, y_pred, y_score)
        m['Fold'] = fold
        all_fold_metrics.append(m)
        print_fold_report(fold, m)

    print("\n" + "=" * 65)
    print("      FINAL 5-FOLD FULL MAIN MODEL SUMMARY (MEAN ± STD)")
    print("=" * 65)
    df_metrics = pd.DataFrame(all_fold_metrics)
    for col in [c for c in df_metrics.columns if c not in ['CM', 'Fold']]:
        mean_v = df_metrics[col].mean()
        std_v = df_metrics[col].std()
        if col in ['TP', 'FP', 'TN', 'FN']:
            print(f"  {col:<13} : {mean_v:.1f} ± {std_v:.1f} (counts)")
        else:
            print(f"  {col:<13} : {mean_v:.2f}% ± {std_v:.2f}%")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()

