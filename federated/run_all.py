import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import torch
import gc
import traceback

from federated.config import FLConfig
from federated.run_single import run_federated_learning
from federated.model_registry import ABLATION_MODELS


def run_all_ablations(config=None):
    """
    Run Federated Learning on all 11 ablation models sequentially.
    Produces a comprehensive comparison summary.
    """
    if config is None:
        config = FLConfig()
    
    print("#" * 70)
    print("#  FEDERATED LEARNING - ALL ABLATION MODELS")
    print(f"#  {len(ABLATION_MODELS)} models × {config.NUM_ROUNDS} rounds × {config.NUM_CLIENTS} clients")
    print("#" * 70)
    print()
    
    all_results = {}
    model_ids = sorted(ABLATION_MODELS.keys(), key=lambda x: int(x[1:]))  # A0, A1, ..., A10
    
    total_start = time.time()
    
    for idx, ablation_id in enumerate(model_ids):
        print(f"\n{'='*70}")
        print(f"  [{idx+1}/{len(model_ids)}] Running {ablation_id}: {ABLATION_MODELS[ablation_id]['name']}")
        print(f"{'='*70}\n")
        
        try:
            result = run_federated_learning(ablation_id, config)
            all_results[ablation_id] = result
        except Exception as e:
            print(f"  *** ERROR running {ablation_id}: {e}")
            traceback.print_exc()
            all_results[ablation_id] = {'error': str(e)}
        
        # Clear GPU memory between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    total_time = time.time() - total_start
    
    # Print summary table
    print("\n" + "#" * 70)
    print("#  SUMMARY: FEDERATED LEARNING ABLATION STUDY")
    print("#" * 70)
    print(f"\n{'ID':<6} {'Model Name':<30} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'Loss':>10}")
    print("-" * 96)
    
    for ablation_id in model_ids:
        result = all_results.get(ablation_id, {})
        if 'error' in result:
            print(f"{ablation_id:<6} {'ERROR':<30} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            continue
        
        final = result.get('final_metrics', {})
        name = result.get('model_name', 'Unknown')[:28]
        acc = f"{final.get('global_accuracy', 0):.4f}"
        f1 = f"{final.get('f1', 0):.4f}"
        prec = f"{final.get('precision', 0):.4f}"
        rec = f"{final.get('recall', 0):.4f}"
        loss = f"{final.get('global_loss', 0):.4f}"
        
        print(f"{ablation_id:<6} {name:<30} {acc:>10} {f1:>10} {prec:>10} {rec:>10} {loss:>10}")
    
    print("-" * 96)
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Save all results
    os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
    output_path = os.path.join(os.path.dirname(__file__), 'results/all_ablations_results.json')
    with open(output_path, 'w') as f:
        json.dump({
            'config': {
                'num_clients': config.NUM_CLIENTS,
                'num_rounds': config.NUM_ROUNDS,
                'local_epochs': config.LOCAL_EPOCHS,
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE,
            },
            'total_time_seconds': total_time,
            'results': all_results
        }, f, indent=2, default=str)
    
    print(f"\nAll results saved to {output_path}")
    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run FL on all ablation models')
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
    
    run_all_ablations(config)
