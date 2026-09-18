import torch
import torch.nn as nn


class LogicEncoder(nn.Module):

    def __init__(
            self,
            input_dim=5,
            embed_dim=768
    ):

        super().__init__()


        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                128
            ),

            nn.GELU(),

            nn.Dropout(0.2),


            nn.Linear(
                128,
                embed_dim
            )

        )


    def forward(self,x):

        return self.encoder(x)