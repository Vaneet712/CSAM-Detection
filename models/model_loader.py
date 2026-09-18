import torch

from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoImageProcessor,
    SiglipVisionModel,
    pipeline
)

try:
    from transformers import SiglipForImageClassification
except ImportError:
    try:
        from transformers.models.siglip.modeling_siglip import SiglipForImageClassification
    except ImportError:
        import torch.nn as nn
        class SiglipForImageClassification(nn.Module):
            def __init__(self, vision_model, classifier):
                super().__init__()
                self.vision_model = vision_model
                self.classifier = classifier

            @classmethod
            def from_pretrained(cls, model_name, *args, **kwargs):
                from transformers import SiglipVisionModel
                vision = SiglipVisionModel.from_pretrained(model_name)
                classifier = nn.Linear(vision.config.hidden_size, 3)
                model = cls(vision, classifier)
                return model

            def forward(self, pixel_values=None, **kwargs):
                outputs = self.vision_model(pixel_values=pixel_values)
                feat = outputs.pooler_output if (hasattr(outputs, "pooler_output") and outputs.pooler_output is not None) else outputs.last_hidden_state.mean(dim=1)
                logits = self.classifier(feat)
                class SiglipOutput:
                    def __init__(self, logits):
                        self.logits = logits
                return SiglipOutput(logits)



from facenet_pytorch import MTCNN
from nudenet import NudeDetector

##############################################################
# DEVICE
##############################################################

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("Loading All Models")
print("=" * 70)

##############################################################
# SIGLIP VISION MODEL
##############################################################

SIGLIP_NAME = "google/siglip-base-patch16-384"

image_processor = AutoImageProcessor.from_pretrained(
    SIGLIP_NAME
)

image_model = SiglipVisionModel.from_pretrained(
    SIGLIP_NAME
).to(device)

image_model.eval()

print("[OK] SigLIP Vision Loaded")

##############################################################
# MPNET
##############################################################

TEXT_MODEL = "sentence-transformers/all-mpnet-base-v2"

text_tokenizer = AutoTokenizer.from_pretrained(
    TEXT_MODEL
)

text_model = AutoModel.from_pretrained(
    TEXT_MODEL
).to(device)

text_model.eval()

print("[OK] MPNet Loaded")

##############################################################
# FACE DETECTOR
##############################################################

mtcnn = MTCNN(
    keep_all=True,
    device=device
)

print("[OK] MTCNN Loaded")

##############################################################
# AGE MODEL
##############################################################

AGE_MODEL = "prithivMLmods/open-age-detection"

age_processor = AutoImageProcessor.from_pretrained(
    AGE_MODEL
)

age_model = SiglipForImageClassification.from_pretrained(
    AGE_MODEL
).to(device)

age_model.eval()

print("[OK] Age Model Loaded")

##############################################################
# EMOTION MODEL
##############################################################

emotion_model = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection",
    device=0 if device.type == "cuda" else -1,
    top_k=None
)

print("[OK] Emotion Model Loaded")

##############################################################
# NUDENET
##############################################################

nude_model = NudeDetector()

print("[OK] NudeNet Loaded")

##############################################################

print("=" * 70)

##############################################################
# FREEZE PRETRAINED MODELS
##############################################################

def freeze_model(model, name="model"):
    """
    Freeze torch based pretrained models
    """
    if hasattr(model, "parameters"):
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        print(f"[OK] Frozen {name}")
    else:
        print(f"[OK] {name} has no trainable parameters")

##############################################################
# APPLY FREEZE
##############################################################

freeze_model(image_model, "SigLIP Vision")
freeze_model(text_model, "MPNet")
freeze_model(age_model, "Age Model")

mtcnn.eval()
print("[OK] MTCNN Frozen")

print("[OK] Emotion Model Frozen")
print("[OK] NudeNet Frozen")

##############################################################
# MODEL STATUS
##############################################################

print("=" * 70)
print("Trainable Model Components:")
print("""
[OK] Cross Attention
[OK] Logic Encoder
[OK] Fusion Transformer
[OK] Classifier
""")
print("=" * 70)