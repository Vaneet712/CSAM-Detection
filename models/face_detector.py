import torch
from PIL import Image

from models.model_loader import mtcnn


@torch.no_grad()
def detect_faces(
    image: Image.Image,
    padding=30,
    min_face_size=20
):
    """
    Detect faces and return padded face crops.

    Parameters
    ----------
    image : PIL.Image

    padding : int
        Extra pixels around face bounding box

    min_face_size : int
        Ignore very small detections


    Returns
    -------
    face_crops : list[PIL.Image]

    boxes : list
        Original face bounding boxes
    """

    try:
        boxes, probs = mtcnn.detect(image)
    except Exception:
        return [], []

    if boxes is None:
        return [], []


    width, height = image.size

    face_crops = []
    final_boxes = []


    for i, box in enumerate(boxes):

        # Confidence check
        if probs is not None:
            if probs[i] < 0.90:
                continue


        x1, y1, x2, y2 = box


        # Convert to int
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)


        # Face size filtering
        face_w = x2 - x1
        face_h = y2 - y1

        if face_w < min_face_size or face_h < min_face_size:
            continue



        # ------------------------------------------------
        # Add padding around face
        # ------------------------------------------------

        x1_pad = max(
            0,
            x1 - padding
        )

        y1_pad = max(
            0,
            y1 - padding
        )

        x2_pad = min(
            width,
            x2 + padding
        )

        y2_pad = min(
            height,
            y2 + padding
        )


        # Crop padded face
        crop = image.crop(
            (
                x1_pad,
                y1_pad,
                x2_pad,
                y2_pad
            )
        )


        face_crops.append(crop)


        # Store original box
        final_boxes.append(
            [
                x1,
                y1,
                x2,
                y2
            ]
        )


    return face_crops, final_boxes