import torch
from transformers import AutoImageProcessor
from facenet_pytorch import MTCNN
from PIL import Image

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

# 1. Setup Device & Load Models
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model_dtype = torch.float16 if device == "cuda" else torch.float32

print("[1/2] Loading Face Detector (MTCNN) and Age Classifier...")
# MTCNN detects and crops all faces in an image
mtcnn = MTCNN(keep_all=True, device=device)

# Hugging Face SigLIP 2 Age Model
MODEL_NAME = "prithivMLmods/open-age-detection"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = SiglipForImageClassification.from_pretrained(
    MODEL_NAME
).to(device).eval()


# 2. Multi-Face Child Detection Function
def check_child_presence(image_path, child_threshold=50.0):
    image = Image.open(image_path).convert("RGB")

    # A. Detect all faces in the image
    boxes, _ = mtcnn.detect(image)

    if boxes is None or len(boxes) == 0:
        return {
            "faces_found": 0,
            "child_present": False,
            "max_child_prob": 0.0,
            "message": "No faces detected in the image."
        }

    print(f"✓ Detected {len(boxes)} face(s) in the image. Analyzing each face...")

    face_probabilities = []
    
    # B. Crop and evaluate each detected face individually
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(b) for b in box]
        # Add margin boundary safety
        x1, y1 = max(0, x1), max(0, y1)
        
        # Crop face from PIL image
        cropped_face = image.crop((x1, y1, x2, y2))

        # Preprocess cropped face
        inputs = processor(images=cropped_face, return_tensors="pt").to(device, dtype=model_dtype)

        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        # Calculate Minor Probability = Child (0-12) + Teenager (13-20)
        p_child = probs[0].item() * 100
        p_teen = probs[1].item() * 100
        p_minor_total = p_child + p_teen

        face_probabilities.append({
            "face_index": idx + 1,
            "child_prob": round(p_minor_total, 2),
            "p_child_0_12": round(p_child, 2),
            "p_teen_13_20": round(p_teen, 2)
        })

    # C. Calculate Maximum Child Probability across all faces
    max_child_probability = max(f["child_prob"] for f in face_probabilities)
    is_child_present = max_child_probability >= child_threshold

    return {
        "faces_found": len(boxes),
        "child_present": is_child_present,
        "max_child_prob_percent": max_child_probability,
        "per_face_breakdown": face_probabilities
    }


# 3. Test Execution
if __name__ == "__main__":
    # Install dependencies if not present:
    # pip install facenet-pytorch pillow transformers torch
    
    IMAGE_PATH = "1.jpeg"  # Image with 2 or 3 people
    
    result = check_child_presence(IMAGE_PATH, child_threshold=50.0)

    print("\n" + "=" * 50)
    print(f"RESULT: Child Present? -> {result['child_present']}")
    print(f"Max Child Probability: {result['max_child_prob_percent']}%")
    print("=" * 50)

    if result["faces_found"] > 0:
        print("\nIndividual Face Evaluation:")
        for f in result["per_face_breakdown"]:
            print(f"  • Face #{f['face_index']}: {f['child_prob']}% probability of being minor/child "
                  f"(Child 0-12: {f['p_child_0_12']}%, Teen 13-20: {f['p_teen_13_20']}%)")