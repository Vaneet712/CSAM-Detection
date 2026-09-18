import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
import torch
from federated.model_registry import get_trainable_state_dict, set_trainable_state_dict


class FedServer:
    """
    Federated Learning Server implementing FedAvg.
    Manages global model and performs weighted parameter aggregation.
    """
    
    def __init__(self, global_model, device):
        self.global_model = global_model
        self.device = device
        self.round_history = []
    
    def get_global_params(self):
        """Get current global trainable parameters."""
        return get_trainable_state_dict(self.global_model)
    
    def distribute_params(self, model):
        """Load global parameters into a client model."""
        global_params = self.get_global_params()
        set_trainable_state_dict(model, global_params)
    
    def aggregate(self, client_params_list, client_weights_list):
        """
        FedAvg aggregation: Weighted average of client parameters.
        
        Args:
            client_params_list: List of state_dicts from clients
            client_weights_list: List of num_samples from each client
            
        Formula: w_global = sum(n_k / N * w_k) for all clients k
        where N = sum(n_k)
        """
        if not client_params_list:
            return
        
        total_samples = sum(client_weights_list)
        
        # Initialize aggregated params with zeros
        aggregated = {}
        for key in client_params_list[0].keys():
            aggregated[key] = torch.zeros_like(client_params_list[0][key])
        
        # Weighted sum
        for params, n_k in zip(client_params_list, client_weights_list):
            weight = n_k / total_samples
            for key in aggregated:
                aggregated[key] += weight * params[key]
        
        # Update global model
        set_trainable_state_dict(self.global_model, aggregated)
        
        return aggregated
    
    def evaluate_global(self, clients, use_val=False):
        """
        Evaluate the global model across all clients' data.
        Returns aggregated metrics including global ROC-AUC and confusion matrix.
        """
        self.global_model.eval()
        
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        client_results = []
        
        for client in clients:
            loader = client.val_loader if use_val and client.val_loader is not None else client.test_loader
            result = client.evaluate(self.global_model, dataloader=loader)
            client_results.append(result)
            
            total_correct += result['accuracy'] * result['num_samples']
            total_samples += result['num_samples']
            total_loss += result['loss'] * result['num_samples']
            all_preds.extend(result['predictions'])
            all_probs.extend(result['probabilities'])
            all_labels.extend(result['labels'])
        
        # Compute metrics
        global_accuracy = total_correct / max(total_samples, 1)
        global_loss = total_loss / max(total_samples, 1)
        
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
        tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        from sklearn.metrics import roc_auc_score, confusion_matrix
        try:
            if len(set(all_labels)) > 1:
                global_roc_auc = float(roc_auc_score(all_labels, all_probs))
            else:
                global_roc_auc = 0.5
        except Exception:
            global_roc_auc = 0.5
            
        global_cm = confusion_matrix(all_labels, all_preds, labels=[0, 1]).tolist()
        
        metrics = {
            'accuracy': global_accuracy,
            'loss': global_loss,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': global_roc_auc,
            'confusion_matrix': global_cm,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'total_samples': total_samples,
            'predictions': all_preds,
            'probabilities': all_probs,
            'labels': all_labels,
            'client_results': client_results
        }
        
        return metrics

