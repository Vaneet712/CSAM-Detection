import torch

from models.model_loader import (
    text_model,
    text_tokenizer,
    device
)


@torch.no_grad()
def encode_text(texts):
    """
    Parameters
    ----------
    texts : list[str]

    Returns
    -------
    word_tokens : Tensor
        (B, Nt, 768)

    attention_mask : Tensor
        (B, Nt)

    cls_token : Tensor
        (B,768)
    """

    inputs = text_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )

    # (B,Nt,768)
    word_tokens = outputs.last_hidden_state

    # CLS token
    cls_token = word_tokens[:, 0]

    return word_tokens, attention_mask, cls_token