import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.optim import AdamW
from training.loss import CSAMLoss
from federated.model_registry import forward_pass, get_trainable_state_dict, set_trainable_state_dict


import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from tqdm import tqdm
from training.loss import CSAMLoss
from federated.model_registry import forward_pass, get_trainable_state_dict, set_trainable_state_dict


class FedClient:
    """
    Federated Learning Client.
    Each client has its own local dataset splits (train, val, test) and performs local training.
    """
    
    def __init__(self, client_id, dataloaders, forward_type, device, lr=1e-4, csam_weight=2.0):
        self.client_id = client_id
        if isinstance(dataloaders, dict):
            self.train_loader = dataloaders.get('train')
            self.val_loader = dataloaders.get('val')
            self.test_loader = dataloaders.get('test')
        else:
            self.train_loader = dataloaders
            self.val_loader = None
            self.test_loader = dataloaders
            
        self.forward_type = forward_type
        self.device = device
        self.lr = lr
        self.criterion = CSAMLoss(csam_weight=csam_weight)
        self.num_samples = len(self.train_loader.dataset) if self.train_loader is not None else 0
    
    def local_train(self, model, local_epochs, patience=3):
        """
        Train the model locally on client train split for up to `local_epochs` epochs with early stopping.
        """
        model.train()
        
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr
        )
        
        best_val_loss = float('inf')
        best_params = get_trainable_state_dict(model)
        patience_counter = 0
        executed_epochs = 0
        
        for epoch in range(local_epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(self.train_loader, desc=f"Client {self.client_id} | Epoch {epoch+1}/{local_epochs}", leave=False)
            for batch in pbar:
                optimizer.zero_grad()
                
                logits, labels = forward_pass(
                    model, batch, self.forward_type, self.device
                )
                
                loss = self.criterion(logits, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            executed_epochs += 1

            # Early stopping evaluation on validation set if available
            eval_loader = self.val_loader if self.val_loader is not None else self.train_loader
            val_eval = self.evaluate(model, dataloader=eval_loader)
            current_val_loss = val_eval['loss']

            print(f"  [Client {self.client_id}] Epoch {epoch+1}/{local_epochs} -> Train Loss: {avg_epoch_loss:.4f} | Val Loss: {current_val_loss:.4f} | Val F1: {val_eval['f1']:.4f}")

            if current_val_loss < best_val_loss - 1e-4:
                best_val_loss = current_val_loss
                best_params = get_trainable_state_dict(model)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience > 0 and patience_counter >= patience:
                    print(f"  [Client {self.client_id}] Early stopping triggered at local epoch {epoch + 1}/{local_epochs} (No val loss improvement for {patience} epochs)")
                    break
        
        set_trainable_state_dict(model, best_params)
        print(f"  ➜ Client {self.client_id} Completed: Trained {executed_epochs}/{local_epochs} local epochs | Best Val Loss={best_val_loss:.4f} | Samples={self.num_samples}\n")
        return best_params, self.num_samples, best_val_loss
    
    def evaluate(self, model, dataloader=None):
        """
        Evaluate the model on local dataset.
        Returns accuracy, loss, precision, recall, f1, roc_auc, confusion_matrix, predictions, probabilities, labels.
        """
        if dataloader is None:
            dataloader = self.test_loader if self.test_loader is not None else self.train_loader
            
        model.eval()
        
        all_preds = []
        all_probs = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Client {self.client_id} [Eval]", leave=False)
            for batch in pbar:
                logits, labels = forward_pass(
                    model, batch, self.forward_type, self.device
                )
                
                loss = self.criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                total_loss += loss.item() * len(labels)
                num_batches += len(labels)
        
        avg_loss = total_loss / max(num_batches, 1)
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        accuracy = correct / max(len(all_preds), 1)
        
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
        tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        try:
            if len(set(all_labels)) > 1:
                roc_auc = float(roc_auc_score(all_labels, all_probs))
            else:
                roc_auc = 0.5
        except Exception:
            roc_auc = 0.5
            
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1]).tolist()
        
        return {
            'client_id': self.client_id,
            'accuracy': accuracy,
            'loss': avg_loss,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'predictions': all_preds,
            'probabilities': all_probs,
            'labels': all_labels,
            'num_samples': len(all_preds)
        }

