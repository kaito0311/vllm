
from typing import cast, Annotated, Literal, TypeAlias

import torch 
import torch.nn as nn
from transformers import ProcessorMixin

from vllm.config import VllmConfig
from vllm.transformers_utils.configs.nanovlm import NanoVLMConfig
from vllm.model_executor.models.utils import IntermediateTensors, maybe_prefix
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import BaseProcessingInfo, BaseDummyInputsBuilder, BaseMultiModalProcessor
from .interfaces import SupportsMultiModal

from .interfaces import MultiModalEmbeddings

from .nano_vlm_modules.language_model import LanguageModel
from .nano_vlm_modules.vision_transformer import ViT
from .nano_vlm_modules.modality_projector import ModalityProjector


class NanoVLMImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - bnp: Batch size * number of images * number of patches
        - c: Number of channels (3)
        - h: Height
        - w: Width
    """

    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, TensorShape("bnp", 3, "h", "w")]
    pixel_attention_mask: Annotated[torch.Tensor, TensorShape("bnp", "h", "w")]
    num_patches: Annotated[torch.Tensor, TensorShape("bn")]


class NanoVLMImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - f: Image feature size
        - h: Hidden size (must match the hidden size of language model backbone)
    """

    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor, TensorShape("bn", "f", "h")]

ImageInputs: TypeAlias = NanoVLMImagePixelInputs | NanoVLMImageEmbeddingInputs


class NanoVLMProcessor(ProcessorMixin):
    # TODO
    ...


class NanoVLMProcessingInfo(BaseProcessingInfo): 
    def get_hf_processor(self, **kwargs: object) -> NanoVLMProcessor:
        ... 
    
class NanoVLMDummyInputsBuilder(BaseDummyInputsBuilder[NanoVLMProcessingInfo]):
    ...

class NanoVLMMultiModalProcessor(BaseMultiModalProcessor[NanoVLMProcessingInfo]):
    # TODO
    ...


class NanoVLMModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: NanoVLMConfig = cast(NanoVLMConfig, vllm_config.model_config.hf_config)

        self.config = config

        self.vision_encoder = ViT(config, prefix=maybe_prefix(prefix, "vision_encoder"))

        self.decoder = LanguageModel(
            config, 
            prefix=maybe_prefix(prefix, "decoder"),
        )

        self.MP = ModalityProjector(
            config,
            prefix=maybe_prefix(prefix, "modality_projector"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.token_embedding(input_ids)

    def _parse_and_validate_image_input(self, **kwargs: object) -> ImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if image_embeds is not None:
            return NanoVLMImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        if pixel_values is not None:
            pixel_attention_mask = kwargs.pop("pixel_attention_mask")
            num_patches = kwargs.pop("num_patches")
            expected_h = expected_w = self.config.vision_config.image_size

            return NanoVLMImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                num_patches=num_patches,
                resolve_bindings={"h": expected_h, "w": expected_w},
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_pixels(self, inputs: NanoVLMImagePixelInputs) -> torch.Tensor:
        pixel_values = inputs["pixel_values"]
        pixel_attention_mask = inputs["pixel_attention_mask"]

        # TODO
        raise NotImplementedError()
        return self.vision_encoder.image_pixels_to_features(
            pixel_values,
            pixel_attention_mask=pixel_attention_mask,
        )

    def _process_image_input(
        self,
        image_input: ImageInputs,
    ) -> torch.Tensor | list[torch.Tensor]:
        if image_input["type"] == "image_embeds":
            return image_input["data"]
        
        raise NotImplementedError()

        # TODO
        image_features = self._process_image_pixels(image_input)
        image_features = self.MP(image_features)

        num_patches = image_input["num_patches"]
        return [e.flatten(0, 1) for e in image_features.split(num_patches.tolist())]

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
       ...



@MULTIMODAL_REGISTRY.register_processor(
    NanoVLMMultiModalProcessor,
    info=NanoVLMProcessingInfo,
    dummy_inputs=NanoVLMDummyInputsBuilder,
)
class NanoVLMForConditionalGeneration(nn.Module, SupportsMultiModal):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # TODO
        raise NotImplementedError()

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        # TODO
        raise NotImplementedError()
    
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # TODO
        raise NotImplementedError()
    
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError()
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

