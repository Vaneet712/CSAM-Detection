import math

import torch
import torch.nn as nn


class CrossAttention(nn.Module):

    def __init__(
        self,
        embed_dim=768,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0

        self.q_proj = nn.Linear(embed_dim, embed_dim)

        self.k_proj = nn.Linear(embed_dim, embed_dim)

        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        image_tokens,
        text_tokens,
        attention_mask=None
    ):

        B = image_tokens.size(0)

        N_img = image_tokens.size(1)

        N_txt = text_tokens.size(1)

        Q = self.q_proj(image_tokens)

        K = self.k_proj(text_tokens)

        V = self.v_proj(text_tokens)

        Q = Q.view(
            B,
            N_img,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        K = K.view(
            B,
            N_txt,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        V = V.view(
            B,
            N_txt,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        scores = torch.matmul(
            Q,
            K.transpose(-2,-1)
        )

        scores = scores / math.sqrt(self.head_dim)

        if attention_mask is not None:

            mask = attention_mask.unsqueeze(1).unsqueeze(2)

            scores = scores.masked_fill(mask==0,-1e9)

        attention = torch.softmax(scores,dim=-1)

        attention = self.dropout(attention)

        out = torch.matmul(attention,V)

        out = out.transpose(1,2)

        out = out.contiguous().view(
            B,
            N_img,
            self.embed_dim
        )

        out = self.out_proj(out)

        out = self.norm(
            out + image_tokens
        )

        return out, attention