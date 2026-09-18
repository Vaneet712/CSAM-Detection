import torch


# Risk vocabulary
BULLY_WORDS = {

    # distress
    "cry":0.5,
    "crying":0.5,
    "scared":0.7,
    "terrified":0.8,
    "afraid":0.6,
    "distressed":0.8,


    # coercion
    "secret":0.8,
    "keep":0.4,
    "don't tell":0.9,
    "forced":0.9,
    "threat":0.9,


    # violence
    "hurt":0.8,
    "abuse":1.0,
    "attack":0.8,


    # child related
    "child":0.4,
    "kid":0.4,
    "minor":0.8
}



def compute_bully_score(text):

    """
    Input:
        text string

    Output:
        bully_score 0-1
    """

    text = text.lower()


    score = 0.0
    count = 0


    for word, weight in BULLY_WORDS.items():

        if word in text:

            score += weight
            count += 1



    if count == 0:
        return 0.0


    score = score / count


    return min(
        score,
        1.0
    )