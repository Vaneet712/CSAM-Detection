import sys
import os
import importlib
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.full_pipeline import FullCSAMPipeline

MODEL_REGISTRY = {
    'A0': {
        'name': 'Full CSAM Model',
        'mode': 'full',
        'forward_type': 'full',
    }
}

def load_model(ablation_id='A0', device='cpu'):
    """
    Loads the Main CSAM Model (FullCSAMPipeline) directly from models.full_pipeline.
    """
    mode = MODEL_REGISTRY.get(ablation_id, {}).get('mode', 'full')
    model = FullCSAMPipeline(mode=mode).to(device)
    return model

def forward_pass(model, batch, forward_type, device):
    images = batch.get('images') if 'images' in batch else batch.get('image')
    if images is not None and isinstance(images, torch.Tensor):
        images = images.to(device)
        
    texts = batch.get('texts') if 'texts' in batch else batch.get('text')
    
    logic_vectors = batch.get('logic_vectors') if 'logic_vectors' in batch else batch.get('logic_vector')
    if logic_vectors is not None and isinstance(logic_vectors, torch.Tensor):
        logic_vectors = logic_vectors.to(device)
        
    labels = batch.get('labels') if 'labels' in batch else batch.get('label')
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.to(device)
        
    if forward_type == 'full':
        output = model(images, texts, logic_vectors)
    elif forward_type == 'no_text':
        output = model(images, logic_vectors)
    elif forward_type == 'no_logic':
        output = model(images, texts)
    elif forward_type == 'no_image':
        output = model(texts, logic_vectors)
    else:
        raise ValueError(f"Unknown forward_type: {forward_type}")
    
    # Handle tuple returns: (logits, attention) or just logits
    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output
        
    return logits, labels

def get_trainable_state_dict(model):
    return {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}

def set_trainable_state_dict(model, state_dict):
    model_dict = dict(model.named_parameters())
    for name, param_data in state_dict.items():
        if name in model_dict:
            model_dict[name].data.copy_(param_data)

def list_models():
    print("Available Models:")
    for key, val in MODEL_REGISTRY.items():
        print(f"{key}: {val['name']} (Forward Type: {val['forward_type']})")
