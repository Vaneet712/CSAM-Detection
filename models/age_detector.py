import torch

from PIL import Image

from models.model_loader import (
    age_model,
    age_processor,
    device
)


@torch.no_grad()
def predict_age(face_image: Image.Image):
    """
    Predict age probability vector.

    Returns
    -------
    [
        child,
        teen,
        adult
    ]
    """

    inputs = age_processor(
        images=face_image,
        return_tensors="pt"
    )


    pixel_values = inputs["pixel_values"].to(device)


    outputs = age_model(
        pixel_values=pixel_values
    )


    logits = outputs.logits


    probs = torch.softmax(
        logits,
        dim=-1
    )[0]


    probs = probs.cpu().numpy()


    # Assuming model classes:
    # 0 -> child
    # 1 -> teen
    # 2 -> adult

    child = float(probs[0])

    teen = float(probs[1])


    if len(probs) >= 3:

        adult = float(probs[2])

    else:

        adult = max(
            0.0,
            1.0 - child - teen
        )


    return [
        child,
        teen,
        adult
    ]