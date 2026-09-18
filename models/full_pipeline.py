import torch
import torch.nn as nn

from models.model_loader import (
    image_model,
    image_processor,
    text_model,
    text_tokenizer,
    device
)

from models.csam_detector import CSAMDetector


class FullCSAMPipeline(nn.Module):

    def __init__(self, mode="full"):

        super().__init__()

        # ============================================================
        # DEFAULT SETTINGS
        # ============================================================

        use_logic = True
        use_cross_attention = True
        use_fusion = True

        modality = "full"

        # ============================================================
        # ABLATION MODES
        # ============================================================

        if mode == "full":

            use_logic = True
            use_cross_attention = True
            use_fusion = True
            modality = "full"

        elif mode == "image_only":

            use_logic = False
            use_cross_attention = False
            use_fusion = True
            modality = "image_only"

        elif mode == "text_only":

            use_logic = False
            use_cross_attention = False
            use_fusion = True
            modality = "text_only"

        elif mode == "no_logic":

            use_logic = False
            use_cross_attention = True
            use_fusion = True
            modality = "full"

        elif mode == "no_cross_attention":

            use_logic = True
            use_cross_attention = False
            use_fusion = True
            modality = "full"

        elif mode == "no_fusion":

            use_logic = True
            use_cross_attention = True
            use_fusion = False
            modality = "full"

        else:

            raise ValueError(
                f"Unknown mode: {mode}"
            )

        # ============================================================
        # DETECTOR
        # ============================================================

        self.detector = CSAMDetector(
            use_logic=use_logic,
            use_cross_attention=use_cross_attention,
            use_fusion=use_fusion,
            modality=modality
        ).to(device)

        # ============================================================
        # FROZEN PRETRAINED MODELS
        # ============================================================

        self.image_model = image_model
        self.text_model = text_model

        self.image_processor = image_processor
        self.text_tokenizer = text_tokenizer

        # ============================================================
        # SAVE MODE
        # ============================================================

        self.mode = mode

    # ================================================================
    # IMAGE ENCODER
    # ================================================================

    def encode_image(self, images):

        if isinstance(images, torch.Tensor):
            pixel_values = images.to(device)
        else:
            inputs = self.image_processor(
                images=list(images),
                return_tensors="pt"
            )
            pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():

            output = self.image_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True
            )

        return output.last_hidden_state

    # ================================================================
    # TEXT ENCODER
    # ================================================================

    def encode_text(self, texts):

        tokens = self.text_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        tokens = {
            k: v.to(device)
            for k, v in tokens.items()
        }

        with torch.no_grad():

            output = self.text_model(
                **tokens
            )

        return output.last_hidden_state

    # ================================================================
    # FORWARD
    # ================================================================

    def forward(
        self,
        images,
        texts,
        logic_vector
    ):

        # Always encode both.
        # CSAMDetector decides which modality to use.

        image_tokens = self.encode_image(
            images
        )

        text_tokens = self.encode_text(
            texts
        )

        # ============================================================
        # DETECTOR
        # ============================================================

        logits, attention = self.detector(
            image_tokens,
            text_tokens,
            logic_vector
        )

        return logits, attention