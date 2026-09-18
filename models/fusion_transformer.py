import torch
import torch.nn as nn



class FusionTransformer(nn.Module):

    def __init__(
            self,
            embed_dim=768,
            num_heads=8,
            num_layers=1
    ):

        super().__init__()


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=0.2,
            batch_first=True
        )


        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )



    def forward(
            self,
            vision_tokens,
            logic_embedding
    ):

        """
        vision_tokens:
            (B,576,768)

        logic_embedding:
            (B,768)

        """

        # convert logic to token

        logic_token = logic_embedding.unsqueeze(1)


        # concatenate

        fused_tokens = torch.cat(
            [
                vision_tokens,
                logic_token
            ],
            dim=1
        )


        output = self.transformer(
            fused_tokens
        )


        return output