import torch
import numpy as np

def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Computes Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    total = len(y_true)
    if total == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0
        }

    accuracy = float((y_true == y_pred).mean())

    # Binary metrics for Class 1 (CSAM)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }

def evaluate_model(model, data_loader, criterion, device):
    """
    Runs model evaluation over data_loader.
    Returns average loss and dictionary of metrics.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            texts = batch["text"]
            logic_vector = batch["logic_vector"].to(device)
            labels = batch["label"].to(device)

            logits, attention = model(images, texts, logic_vector)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * len(labels)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    avg_loss = total_loss / len(data_loader.dataset) if len(data_loader.dataset) > 0 else 0.0
    metrics = compute_classification_metrics(all_targets, all_preds, all_probs)
    metrics["loss"] = round(avg_loss, 4)

    return metrics, all_preds, all_targets
