import torch
from PIL import Image
from facenet_pytorch import MTCNN
from transformers import AutoImageProcessor, pipeline

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

# 1. Device Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_dtype = torch.float16 if device == "cuda" else torch.float32

print(f"[1/2] Loading Models on {device.upper()}...")

# A. Face Detector (MTCNN)
mtcnn = MTCNN(keep_all=True, device=device)

# B. Age Model (prithivMLmods/open-age-detection)
AGE_MODEL_NAME = "prithivMLmods/open-age-detection"
age_processor = AutoImageProcessor.from_pretrained(AGE_MODEL_NAME)
age_model = SiglipForImageClassification.from_pretrained(
    AGE_MODEL_NAME
).to(device).eval()

# C. Emotion Model (dima806/facial_emotions_image_detection)
emotion_classifier = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection",
    device=0 if device == "cuda" else -1,
    top_k=None  # Get probabilities for ALL classes
)

print("✓ Models loaded successfully!\n")


# 2. Combined Inference Function
def process_image(image_path):
    img = Image.open(image_path).convert("RGB")
    
    # Detect bounding boxes for all faces
    boxes, _ = mtcnn.detect(img)
    if boxes is None or len(boxes) == 0:
        return {"faces_found": 0, "results": []}

    results = []

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [max(0, int(b)) for b in box]
        cropped_face = img.crop((x1, y1, x2, y2))

        # --- STAGE 1: AGE DETECTION ---
        age_inputs = age_processor(images=cropped_face, return_tensors="pt").to(device, dtype=model_dtype)
        with torch.inference_mode():
            age_logits = age_model(**age_inputs).logits
            age_probs = torch.softmax(age_logits, dim=-1)[0]

        p_child = age_probs[0].item() * 100  # Child 0-12
        p_teen  = age_probs[1].item() * 100  # Teen 13-20
        p_minor = p_child + p_teen           # Aggregate Minor Prob

        # --- STAGE 2: EMOTION PROBABILITIES ---
        emotion_outputs = emotion_classifier(cropped_face)
        
        # Sort emotions by confidence score
        emotion_probs = {
            item['label'].lower(): round(item['score'] * 100, 2) 
            for item in sorted(emotion_outputs, key=lambda x: x['score'], reverse=True)
        }

        results.append({
            "face_index": idx + 1,
            "bbox": [x1, y1, x2, y2],
            "age_breakdown": {
                "minor_total_percent": round(p_minor, 2),
                "child_0_12": round(p_child, 2),
                "teen_13_20": round(p_teen, 2),
            },
            "emotion_probabilities_percent": emotion_probs
        })

    return {"faces_found": len(boxes), "results": results}


# 3. Execution Example
if __name__ == "__main__":
    IMAGE_PATH = "1.jpeg"  # Replace with your test image
    
    output = process_image(IMAGE_PATH)
    
    print("=" * 60)
    print(f"Total Faces Found: {output['faces_found']}")
    print("=" * 60)

    for face in output["results"]:
        print(f"\nFace #{face['face_index']} - BBox: {face['bbox']}")
        print(f"  • Minor/Child Probability : {face['age_breakdown']['minor_total_percent']}%")
        print(f"    - Child (0-12)         : {face['age_breakdown']['child_0_12']}%")
        print(f"    - Teen (13-20)         : {face['age_breakdown']['teen_13_20']}%")
        print("  • Emotion Probabilities   :")
        for emo, score in face["emotion_probabilities_percent"].items():
            print(f"    - {emo:<10}: {score}%")