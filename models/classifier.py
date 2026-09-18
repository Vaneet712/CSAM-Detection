import torch.nn as nn



class CSAMClassifier(nn.Module):

    def __init__(
            self,
            embed_dim=768,
            num_classes=2
    ):

        super().__init__()


        self.head = nn.Sequential(

            nn.Linear(
                embed_dim,
                256
            ),

            nn.GELU(),

            nn.Dropout(0.3),


            nn.Linear(
                256,
                num_classes
            )

        )


    def forward(self,x):

        """
        x:
        Fusion output

        (B,577,768)
        """


        # CLS token

        cls_token = x[:,0,:]


        logits = self.head(
            cls_token
        )


        return logits