from PIL import Image
import numpy as np

from models.model_loader import nude_model


def predict_nudity(image):

    """
    Input:
        image path OR PIL image

    Output:
        P_nudity
    """


    # If PIL image is provided
    if isinstance(image, Image.Image):

        image = np.array(image)


    detections = nude_model.detect(
        image
    )


    if len(detections) == 0:
        return 0.0


    max_score = 0.0


    for item in detections:

        score = item.get(
            "score",
            0.0
        )

        max_score = max(
            max_score,
            score
        )


    return float(max_score)