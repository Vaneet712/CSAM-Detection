import torch
import torch.nn as nn

from models.cross_attention import CrossAttention
from models.logic_encoder import LogicEncoder
from models.fusion_transformer import FusionTransformer
from models.classifier import CSAMClassifier


class CSAMDetector(nn.Module):

    def __init__(
        self,
        embed_dim=768,
        use_logic=True,
        use_cross_attention=True,
        use_fusion=True,
        modality="full"
    ):
        super().__init__()

        self.use_logic = use_logic
        self.use_cross_attention = use_cross_attention
        self.use_fusion = use_fusion
        self.modality = modality

        # ============================================================
        # CROSS ATTENTION
        # ============================================================

        self.cross_attention = CrossAttention(
            embed_dim=embed_dim,
            num_heads=8
        )

        # ============================================================
        # LOGIC ENCODER
        # ============================================================

        self.logic_encoder = LogicEncoder(
            input_dim=5,
            embed_dim=embed_dim
        )

        # ============================================================
        # FUSION TRANSFORMER
        # ============================================================

        self.fusion = FusionTransformer(
            embed_dim=embed_dim,
            num_heads=8,
            num_layers=1
        )

        # ============================================================
        # CLASSIFIER
        # ============================================================

        self.classifier = CSAMClassifier(
            embed_dim=embed_dim,
            num_classes=2
        )

    # ================================================================
    # FORWARD
    # ================================================================

    def forward(
        self,
        image_tokens,
        text_tokens,
        logic_vector
    ):

        # ============================================================
        # IMAGE ONLY
        # ============================================================

        if self.modality == "image_only":

            image_global = image_tokens.mean(
                dim=1,
                keepdim=True
            )

            vision_tokens = image_global

            attention_weights = None

        # ============================================================
        # TEXT ONLY
        # ============================================================

        elif self.modality == "text_only":

            text_global = text_tokens.mean(
                dim=1,
                keepdim=True
            )

            vision_tokens = text_global

            attention_weights = None

        # ============================================================
        # FULL / MODULE ABLATIONS
        # ============================================================

        else:

            # --------------------------------------------------------
            # FULL CROSS-ATTENTION
            # --------------------------------------------------------

            if self.use_cross_attention:

                vision_tokens, attention_weights = (
                    self.cross_attention(
                        image_tokens,
                        text_tokens
                    )
                )

            # --------------------------------------------------------
            # NO CROSS-ATTENTION
            # --------------------------------------------------------

            else:

                # Independently pool both modalities.
                # No image-text attention is performed.

                image_global = image_tokens.mean(
                    dim=1,
                    keepdim=True
                )

                text_global = text_tokens.mean(
                    dim=1,
                    keepdim=True
                )

                # Simple modality combination
                combined_global = (
                    image_global + text_global
                ) / 2.0

                vision_tokens = combined_global

                attention_weights = None

        # ============================================================
        # LOGIC BRANCH
        # ============================================================

        if self.use_logic:

            logic_embedding = self.logic_encoder(
                logic_vector
            )

        else:

            batch_size = vision_tokens.size(0)
            embed_dim = vision_tokens.size(2)

            logic_embedding = torch.zeros(
                batch_size,
                embed_dim,
                device=vision_tokens.device,
                dtype=vision_tokens.dtype
            )

        # ============================================================
        # FUSION
        # ============================================================

        if self.use_fusion:

            fused_tokens = self.fusion(
                vision_tokens,
                logic_embedding
            )

        # ============================================================
        # NO FUSION
        # ============================================================

        else:

            # No Fusion Transformer.
            #
            # We create a simple global representation from
            # vision + logic so that BOTH actually contribute
            # to classification.

            vision_global = vision_tokens.mean(
                dim=1
            )

            fused_global = (
                vision_global + logic_embedding
            ) / 2.0

            fused_tokens = fused_global.unsqueeze(1)

        # ============================================================
        # CLASSIFICATION
        # ============================================================

        logits = self.classifier(
            fused_tokens
        )

        return logits, attention_weights