import torch


def evaluate_logic(
        age_vectors,
        emotion_vectors,
        nudity_score,
        bully_score
):

    """
    Creates final logic vector

    Output:

    [
    nudity,
    minor_probability,
    csam_init,
    adult_abuse,
    bully_score
    ]

    """

    p_minor_max = 0.0

    adult_hostility = 0.0

    minor_distress = []


    face_data = []


    for age, emo in zip(
        age_vectors,
        emotion_vectors
    ):


        child, teen, adult = age


        minor = child + teen


        p_minor_max = max(
            p_minor_max,
            minor
        )


        # adult anger/happy

        if adult >= 0.5:

            adult_hostility = max(
                adult_hostility,
                emo[2] + emo[3]
            )


        # child distress

        if minor >= 0.5:

            minor_distress.append(
                emo[0] + emo[1]
            )


        face_data.append(
            {
            "age":age,
            "emotion":emo
            }
        )



    # -----------------------------
    # CSAM Logic
    # -----------------------------


    if p_minor_max >= 0.20:


        r_nudity = (
            p_minor_max *
            nudity_score
        )


        distress = 0.0


        if len(minor_distress)>0:

            distress=max(
                minor_distress
            )


        r_distress=min(
            1.0,
            distress *
            (1+0.5*adult_hostility)
        )


        csam_init = (
            1 -
            (
            (1-r_nudity)
            *
            (1-r_distress)
            )
        )


        adult_abuse = 0.0



    else:


        csam_init=0.0


        adult_distress=max(
            [
            e[0]+e[1]
            for e in emotion_vectors
            ]
        )


        adult_abuse = (
            nudity_score *
            adult_distress
        )



    return torch.tensor(
        [
        nudity_score,
        p_minor_max,
        csam_init,
        adult_abuse,
        bully_score
        ],
        dtype=torch.float32
    )