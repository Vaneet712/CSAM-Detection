import sys
import os
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from federated.config import FLConfig
from federated.data_utils import load_real_client_dataloaders
from federated.model_registry import load_model, get_trainable_state_dict, set_trainable_state_dict
from federated.client import FedClient
from federated.server import FedServer


def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-CSAM', 'CSAM'],
                yticklabels=['Non-CSAM', 'CSAM'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(labels, probs, roc_auc_val, title, save_path):
    plt.figure(figsize=(6, 5))
    if len(set(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.4f})')
    else:
        plt.plot([0, 1], [0, 1], color='darkorange', lw=2, label=f'ROC curve (N/A)')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_federated_learning_main_model(config=None):
    if config is None:
        config = FLConfig()

    print("=" * 80)
    print("FEDERATED LEARNING: MAIN MODEL (FULL CSAM PIPELINE)")
    print("Coordinator/Server: Client 1 Node")
    print(f"Clients: {config.NUM_CLIENTS} | Rounds: {config.NUM_ROUNDS} | Local Epochs: {config.LOCAL_EPOCHS} | Batch Size: {config.BATCH_SIZE} | LR: {config.LEARNING_RATE}")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # 1. Load real client datasets
    print("\n[1/4] Loading real data partitions for 5 clients...")
    client_loaders = load_real_client_dataloaders(
        splits_dir=config.DATA_SPLITS_DIR,
        image_dir=config.IMAGE_DIR,
        batch_size=config.BATCH_SIZE,
        num_clients=config.NUM_CLIENTS
    )
    for cname, loaders in client_loaders.items():
        print(f"  {cname.upper()}: Train={len(loaders['train_ds'])} | Val={len(loaders['val_ds'])} | Test={len(loaders['test_ds'])}")

    # 2. Load global main model (A0 / Full Pipeline)
    print("\n[2/4] Initializing Global Main Model (Full Pipeline)...")
    global_model = load_model('A0', device)

    total_params = sum(p.numel() for p in global_model.parameters())
    trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # 3. Create Server (Coordinator) and Clients
    print(f"\n[3/4] Initializing Server/Coordinator on Client 1 and setup {config.NUM_CLIENTS} Clients...")
    server = FedServer(global_model, device)

    clients = []
    for i in range(1, config.NUM_CLIENTS + 1):
        cname = f"client_{i}"
        client = FedClient(
            client_id=i,
            dataloaders=client_loaders[cname],
            forward_type='full',
            device=device,
            lr=config.LEARNING_RATE,
            csam_weight=config.CSAM_WEIGHT
        )
        clients.append(client)

    # 4. Federated Training Rounds
    print(f"\n[4/4] Starting {config.NUM_ROUNDS} Federated Learning rounds...\n")
    round_history = []
    best_val_f1 = -1.0
    best_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "federated_main_model_best.pt")

    for round_num in range(1, config.NUM_ROUNDS + 1):
        r_start = time.time()
        print(f"--- Round {round_num}/{config.NUM_ROUNDS} ---")

        client_params_list = []
        client_weights_list = []
        client_train_losses = []

        for client in clients:
            # Distribute global model weights from Coordinator
            server.distribute_params(global_model)

            # Local training on client with early stopping patience
            patience = getattr(config, 'EARLY_STOPPING_PATIENCE', 3)
            updated_params, n_samples, avg_loss = client.local_train(global_model, config.LOCAL_EPOCHS, patience=patience)

            client_params_list.append(updated_params)
            client_weights_list.append(n_samples)
            client_train_losses.append(avg_loss)

        # FedAvg Aggregation at Server/Coordinator
        server.aggregate(client_params_list, client_weights_list)

        # Evaluate global model on clients' validation data
        val_metrics = server.evaluate_global(clients, use_val=True)
        r_time = time.time() - r_start

        avg_client_train_loss = sum(client_train_losses) / max(len(client_train_losses), 1)

        print(f"  → Server Aggregation Complete | Avg Train Loss={avg_client_train_loss:.4f}")
        print(f"  → Global Val | Acc={val_metrics['accuracy']:.4f} | Loss={val_metrics['loss']:.4f} | "
              f"F1={val_metrics['f1']:.4f} | ROC-AUC={val_metrics['roc_auc']:.4f} | "
              f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} | Time={r_time:.1f}s\n")

        round_history.append({
            'round': round_num,
            'avg_train_loss': avg_client_train_loss,
            'val_accuracy': val_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_roc_auc': val_metrics['roc_auc'],
            'round_time': r_time
        })

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(server.get_global_params(), best_checkpoint_path)
            print(f"  [SAVED] New best global model saved (Val F1: {best_val_f1:.4f})")

    # Load best global weights
    if os.path.exists(best_checkpoint_path):
        best_params = torch.load(best_checkpoint_path, map_location=device)
        set_trainable_state_dict(global_model, best_params)
        print(f"\n[OK] Loaded best global model checkpoint from {best_checkpoint_path}")

    # Final Save of Federated Model
    final_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "federated_main_model_final.pt")
    torch.save(global_model.state_dict(), final_checkpoint_path)
    print(f"[OK] Saved final complete state dict to {final_checkpoint_path}")

    # =========================================================================
    # FINAL TEST EVALUATION ACROSS ALL CLIENTS & GLOBAL
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL FEDERATED TEST EVALUATION ON ALL CLIENTS")
    print("=" * 80)

    per_client_test_results = []
    
    for i, client in enumerate(clients, start=1):
        c_name = f"client_{i}"
        c_res = client.evaluate(global_model, dataloader=client.test_loader)
        per_client_test_results.append(c_res)

        # Plot Confusion Matrix
        cm_path = os.path.join(config.RESULTS_DIR, f"client_{i}_confusion_matrix.png")
        plot_confusion_matrix(c_res['confusion_matrix'], f"Client {i} Confusion Matrix (Federated Main Model)", cm_path)

        # Plot ROC Curve
        roc_path = os.path.join(config.RESULTS_DIR, f"client_{i}_roc_curve.png")
        plot_roc_curve(c_res['labels'], c_res['probabilities'], c_res['roc_auc'], f"Client {i} ROC Curve (Federated Main Model)", roc_path)

        print(f"\n--- CLIENT {i} TEST METRICS ---")
        print(f"  Samples         : {c_res['num_samples']}")
        print(f"  Accuracy        : {c_res['accuracy']:.4f}")
        print(f"  Loss            : {c_res['loss']:.4f}")
        print(f"  Precision       : {c_res['precision']:.4f}")
        print(f"  Recall          : {c_res['recall']:.4f}")
        print(f"  F1-Score        : {c_res['f1']:.4f}")
        print(f"  ROC-AUC Score   : {c_res['roc_auc']:.4f}")
        print(f"  Confusion Matrix: TN={c_res['tn']}, FP={c_res['fp']}, FN={c_res['fn']}, TP={c_res['tp']}")
        print(f"  [[TN, FP], [FN, TP]] = {c_res['confusion_matrix']}")
        print(f"  Confusion Matrix Plot saved: {cm_path}")
        print(f"  ROC Curve Plot saved       : {roc_path}")

    # Evaluate Combined Global Performance across all 5 Clients
    global_test_res = server.evaluate_global(clients, use_val=False)

    global_cm_path = os.path.join(config.RESULTS_DIR, "global_confusion_matrix.png")
    plot_confusion_matrix(global_test_res['confusion_matrix'], "Global Combined Confusion Matrix (Federated Main Model)", global_cm_path)

    global_roc_path = os.path.join(config.RESULTS_DIR, "global_roc_curve.png")
    plot_roc_curve(global_test_res['labels'], global_test_res['probabilities'], global_test_res['roc_auc'], "Global Combined ROC Curve (Federated Main Model)", global_roc_path)

    print("\n" + "=" * 80)
    print("GLOBAL COMBINED TEST METRICS (ALL 5 CLIENTS)")
    print("=" * 80)
    print(f"  Total Test Samples : {global_test_res['total_samples']}")
    print(f"  Global Accuracy    : {global_test_res['accuracy']:.4f}")
    print(f"  Global Loss        : {global_test_res['loss']:.4f}")
    print(f"  Global Precision   : {global_test_res['precision']:.4f}")
    print(f"  Global Recall      : {global_test_res['recall']:.4f}")
    print(f"  Global F1-Score    : {global_test_res['f1']:.4f}")
    print(f"  Global ROC-AUC     : {global_test_res['roc_auc']:.4f}")
    print(f"  Global Confusion Matrix: TN={global_test_res['tn']}, FP={global_test_res['fp']}, FN={global_test_res['fn']}, TP={global_test_res['tp']}")
    print(f"  [[TN, FP], [FN, TP]]     = {global_test_res['confusion_matrix']}")
    print(f"  Global Confusion Matrix Plot saved: {global_cm_path}")
    print(f"  Global ROC Curve Plot saved       : {global_roc_path}")
    print("=" * 80)

    # Save summary results JSON
    summary_output = {
        'model_name': 'Full CSAM Pipeline (Main Model)',
        'config': {
            'num_clients': config.NUM_CLIENTS,
            'num_rounds': config.NUM_ROUNDS,
            'local_epochs': config.LOCAL_EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
        },
        'per_client_test_metrics': [
            {
                'client_id': r['client_id'],
                'num_samples': r['num_samples'],
                'accuracy': r['accuracy'],
                'loss': r['loss'],
                'precision': r['precision'],
                'recall': r['recall'],
                'f1': r['f1'],
                'roc_auc': r['roc_auc'],
                'confusion_matrix': r['confusion_matrix'],
                'tp': r['tp'], 'fp': r['fp'], 'fn': r['fn'], 'tn': r['tn']
            } for r in per_client_test_results
        ],
        'global_test_metrics': {
            'total_samples': global_test_res['total_samples'],
            'accuracy': global_test_res['accuracy'],
            'loss': global_test_res['loss'],
            'precision': global_test_res['precision'],
            'recall': global_test_res['recall'],
            'f1': global_test_res['f1'],
            'roc_auc': global_test_res['roc_auc'],
            'confusion_matrix': global_test_res['confusion_matrix'],
            'tp': global_test_res['tp'], 'fp': global_test_res['fp'], 'fn': global_test_res['fn'], 'tn': global_test_res['tn']
        },
        'round_history': round_history
    }

    json_output_path = os.path.join(config.RESULTS_DIR, "federated_main_model_results.json")
    with open(json_output_path, 'w') as f:
        json.dump(summary_output, f, indent=2)

    print(f"\n[SUCCESS] Saved complete federated learning results to {json_output_path}\n")
    return summary_output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run Federated Learning on Main Model across 5 Clients')
    parser.add_argument('--rounds', type=int, default=5, help='Number of federated rounds')
    parser.add_argument('--local-epochs', type=int, default=15, help='Maximum local epochs per round')
    parser.add_argument('--early-stopping-patience', type=int, default=3, help='Patience for local early stopping')
    parser.add_argument('--clients', type=int, default=5, help='Number of clients')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()

    config = FLConfig()
    config.NUM_ROUNDS = args.rounds
    config.LOCAL_EPOCHS = args.local_epochs
    config.EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    config.NUM_CLIENTS = args.clients
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr

    run_federated_learning_main_model(config)
