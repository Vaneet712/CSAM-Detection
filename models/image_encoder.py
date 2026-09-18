import torch

from models.model_loader import (
    image_model,
    device
)


@torch.no_grad()
def encode_images(pixel_values):
    """
    Parameters
    ----------
    pixel_values : Tensor
        Shape: (B,3,384,384)

    Returns
    -------
    patch_tokens : Tensor
        Shape: (B,Np,768)

    pool_token : Tensor
        Shape: (B,768)
    """

    pixel_values = pixel_values.to(device)

    outputs = image_model(

        pixel_values=pixel_values,

        output_hidden_states=True,

        return_dict=True

    )

    patch_tokens = outputs.last_hidden_state

    if outputs.pooler_output is not None:

        pool_token = outputs.pooler_output

    else:

        pool_token = patch_tokens.mean(dim=1)

    return patch_tokens, pool_token