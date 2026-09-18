import torch
import torch.nn as nn



class CSAMLoss(nn.Module):

    def __init__(
            self,
            csam_weight=2.0
    ):

        super().__init__()


        self.csam_weight = csam_weight



    def forward(
            self,
            logits,
            labels
    ):


        ############################################
        # Create weights on same device as logits
        ############################################

        weights = torch.tensor(

            [
                1.0,
                self.csam_weight
            ],

            device=logits.device

        )



        loss = nn.functional.cross_entropy(

            logits,

            labels,

            weight=weights

        )


        return loss