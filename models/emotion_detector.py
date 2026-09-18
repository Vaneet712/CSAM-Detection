import torch

from PIL import Image

from models.model_loader import emotion_model


@torch.no_grad()
def predict_emotion(face_image: Image.Image):
    """
    Predict facial emotion.

    Returns
    -------
    [
        fear,
        sad,
        angry,
        happy,
        neutral
    ]
    """


    results = emotion_model(
        face_image
    )


    emotion_map = {}


    for item in results:

        label = item["label"].lower()

        score = item["score"]

        emotion_map[label] = score



    fear = float(
        emotion_map.get(
            "fear",
            0.0
        )
    )


    sad = float(
        emotion_map.get(
            "sad",
            0.0
        )
    )


    angry = float(
        emotion_map.get(
            "angry",
            0.0
        )
    )


    happy = float(
        emotion_map.get(
            "happy",
            0.0
        )
    )


    neutral = float(
        emotion_map.get(
            "neutral",
            0.0
        )
    )


    return [
        fear,
        sad,
        angry,
        happy,
        neutral
    ]