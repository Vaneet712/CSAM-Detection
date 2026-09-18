import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
import time
import torch
import json
import logging

from federated.config import FLConfig
from federated.data_utils import create_client_dataloaders
from federated.model_registry import load_model, ABLATION_MODELS, get_trainable_state_dict, set_trainable_state_dict
from federated.client import FedClient
from federated.server import FedServer


def run_federated_learning(ablation_id, config=None):
    """
    Run Federated Learning with FedAvg for a specific ablation model.
    
    Args:
        ablation_id: str, e.g., 'A0', 'A1', ..., 'A10'
        config: FLConfig instance (uses defaults if None)
    
    Returns:
        dict with training history and final metrics
    """
    if config is None:
        config = FLConfig()
    
    if ablation_id not in ABLATION_MODELS:
        raise ValueError(f"Unknown ablation ID: {ablation_id}. Available: {list(ABLATION_MODELS.keys())}")
        
    model_info = ABLATION_MODELS[ablation_id]
    print("=" * 70)
    print(f"FEDERATED LEARNING: {ablation_id} - {model_info['name']}")
    print(f"Forward type: {model_info['forward_type']}")
    print(f"Clients: {config.NUM_CLIENTS} | Rounds: {config.NUM_ROUNDS} | "
          f"Local Epochs: {config.LOCAL_EPOCHS} | LR: {config.LEARNING_RATE}")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Step 1: Create client dataloaders
    print("\n[1/4] Creating client data partitions...")
    try:
        client_loaders = create_client_dataloaders(
            num_clients=config.NUM_CLIENTS,
            samples_per_client=config.NUM_SAMPLES_PER_CLIENT,
            batch_size=config.BATCH_SIZE,
            iid=config.IID
        )
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        raise
    
    # Step 2: Initialize global model
    print(f"[2/4] Loading model: {model_info['class_name']}...")
    try:
        global_model = load_model(ablation_id, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Count parameters
    total_params = sum(p.numel() for p in global_model.parameters())
    trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    # Step 3: Create server and clients
    print(f"[3/4] Setting up server and {config.NUM_CLIENTS} clients...")
    try:
        server = FedServer(global_model, device)
        
        clients = []
        for i in range(config.NUM_CLIENTS):
            client = FedClient(
                client_id=i,
                dataloader=client_loaders[i],
                forward_type=model_info['forward_type'],
                device=device,
                lr=config.LEARNING_RATE,
                csam_weight=config.CSAM_WEIGHT
            )
            clients.append(client)
    except Exception as e:
        print(f"Error setting up clients/server: {e}")
        raise
    
    # Step 4: Federated training rounds
    print(f"[4/4] Starting {config.NUM_ROUNDS} federated rounds...\n")
    
    history = []
    
    for round_num in range(1, config.NUM_ROUNDS + 1):
        round_start = time.time()
        print(f"--- Round {round_num}/{config.NUM_ROUNDS} ---")
        
        client_params_list = []
        client_weights_list = []
        client_losses = []
        
        try:
            for client in clients:
                # Distribute global params to client
                server.distribute_params(global_model)
                
                # Local training
                updated_params, n_samples, avg_loss = client.local_train(
                    global_model, config.LOCAL_EPOCHS
                )
                
                client_params_list.append(updated_params)
                client_weights_list.append(n_samples)
                client_losses.append(avg_loss)
            
            # FedAvg aggregation
            server.aggregate(client_params_list, client_weights_list)
            
            # Evaluate global model
            metrics = server.evaluate_global(clients)
            
        except Exception as e:
            print(f"Error during federated training round {round_num}: {e}")
            raise
            
        round_time = time.time() - round_start
        
        round_info = {
            'round': round_num,
            'avg_client_loss': sum(client_losses) / len(client_losses) if client_losses else 0,
            'global_accuracy': metrics['accuracy'],
            'global_loss': metrics['loss'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'round_time': round_time
        }
        history.append(round_info)
        
        print(f"  → Global | Acc={metrics['accuracy']:.4f} | "
              f"Loss={metrics['loss']:.4f} | F1={metrics['f1']:.4f} | "
              f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} | "
              f"Time={round_time:.1f}s")
        print()
    
    # Final results
    final = history[-1] if history else {}
    result = {
        'ablation_id': ablation_id,
        'model_name': model_info['name'],
        'forward_type': model_info['forward_type'],
        'config': {
            'num_clients': config.NUM_CLIENTS,
            'num_rounds': config.NUM_ROUNDS,
            'local_epochs': config.LOCAL_EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
        },
        'trainable_params': trainable_params,
        'final_metrics': final,
        'history': history
    }
    
    print("=" * 70)
    print(f"COMPLETED: {ablation_id} - {model_info['name']}")
    print(f"Final Accuracy: {final.get('global_accuracy', 'N/A')}")
    print(f"Final F1: {final.get('f1', 'N/A')}")
    print("=" * 70)
    
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run FL for a single ablation model')
    parser.add_argument('--model', type=str, default='A0', 
                        choices=list(ABLATION_MODELS.keys()),
                        help='Ablation model ID (A0-A10)')
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--local-epochs', type=int, default=2)
    parser.add_argument('--clients', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    config = FLConfig()
    config.NUM_ROUNDS = args.rounds
    config.LOCAL_EPOCHS = args.local_epochs
    config.NUM_CLIENTS = args.clients
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    
    try:
        result = run_federated_learning(args.model, config)
        
        # Save result
        os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
        output_path = os.path.join(os.path.dirname(__file__), f'results/{args.model}_result.json')
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")
    except Exception as e:
        print(f"Failed to run FL for {args.model}: {e}")
        sys.exit(1)
