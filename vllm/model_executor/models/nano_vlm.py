
from typing import Optional, cast, Annotated, Literal, TypeAlias, Union
from itertools import accumulate

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from transformers import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_base import BatchFeature as ImageBatchFeature

from vllm.config import VllmConfig
from vllm.transformers_utils.configs.nano_vlm import NanoVLMConfig
from vllm.model_executor.models.utils import IntermediateTensors, maybe_prefix
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import BaseProcessingInfo, BaseDummyInputsBuilder, BaseMultiModalProcessor
from .interfaces import SupportsMultiModal

from .interfaces import MultiModalEmbeddings

from .nano_vlm_modules.language_model import LanguageModel
from .nano_vlm_modules.vision_transformer import ViT
from .nano_vlm_modules.modality_projector import ModalityProjector
from .nano_vlm_modules.utils.custom_transforms import DynamicResize, GlobalAndSplitImages


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


class NanoVLMImageProcessor:
    def __init__(self, max_image_size: int, patch_size: int, resize_to_max_size_len: bool = False):
        self.max_image_size = max_image_size
        self.patch_size = patch_size
        self.resize_to_max_size_len = resize_to_max_size_len

        self.transforms = transforms.Compose([
            DynamicResize(
                patch_size=self.patch_size,
                max_side_len=self.max_image_size,
                resize_to_max_side_len=self.resize_to_max_size_len,
            ),
            transforms.ToTensor(),
            GlobalAndSplitImages(patch_size),
        ])

    def __call__(self, ls_ls_images: list[list[Image.Image]]) -> ImageBatchFeature:

        processed_images = []
        rows = []
        cols = []

        for ls_images in ls_ls_images:
            row_images = []
            assert len(
                ls_images) == 1, "Currently only support 1 image per row for simplicity; this can be easily relaxed in the future if needed."
            for image in ls_images:
                image_tensor, (num_patches_h,
                               num_patches_w) = self.transforms(image)
                row_images.append(image_tensor)

                rows.append([num_patches_h])
                cols.append([num_patches_w])
                processed_images.append(row_images)

        processed_images = torch.tensor(processed_images)  # (bn, p, c, h, w)
        rows = torch.tensor(rows)  # (bn, 1)
        cols = torch.tensor(cols)  # (bn, 1)

        bn, p, c, h, w = processed_images.shape

        pixel_attention_mask = torch.ones((bn, p, h, w), dtype=torch.bool)

        return ImageBatchFeature(
            data={
                "pixel_values": processed_images,
                "pixel_attention_mask": pixel_attention_mask,
                "rows": rows,
                "cols": cols,
            }
        )


class NanoVLMProcessor(ProcessorMixin):
    # TODO
    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        image_processor,
        tokenizer,
        image_seq_len: int = 64,
        **kwargs
    ):
        self.image_token = "<|image|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.global_image_token = "<|global_image|>"
        self.global_image_token_id = tokenizer.convert_tokens_to_ids(
            self.global_image_token)

        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_seq_len = image_seq_len

    def __call__(
        self,
        images: Union[Image.Image, list[Image.Image],
                      list[list[Image.Image]]] = None,
        text: Union[str, list[str]] = None,
        audio=None,
        videos=None,
        image_seq_len: Optional[int] = None,
        **kwargs,
    ) -> BatchEncoding:
        assert audio is None, "Audio input is not supported."
        assert videos is None, "Video input is not supported."

        image_seq_len = image_seq_len or self.image_seq_len

        n_images_in_text = []
        n_images_in_images = []
        inputs = {}

        if text is not None:
            if isinstance(text, str):
                text = [text]
            if isinstance(text, list):
                if not isinstance(text[0], str):
                    raise ValueError(
                        "All elements in the text list must be strings.")
            n_images_in_text = [sample.count(
                self.image_token) for sample in text]

        ls_ls_images: list[list[Image.Image]] | None = None

        if images is not None:
            if isinstance(images, Image.Image):
                ls_ls_images = [[images]]
            elif isinstance(images, (list, tuple)) and isinstance(images[0], Image.Image):
                images: list[Image.Image]
                if text is not None:
                    if sum(n_images_in_text) != len(images):
                        raise ValueError(
                            f"The total number of {self.image_token} tokens in the prompts should be the same as the number of images passed."
                            f" Found {sum(n_images_in_text)} {self.image_token} tokens and {len(images)} images."
                        )
                    # Reorganize the images to match the prompts
                    cumsum_images_in_text = [0] + \
                        list(accumulate(n_images_in_text))
                    ls_ls_images = [
                        cast(list[Image.Image], images)[
                            cumsum_images_in_text[i]: cumsum_images_in_text[i + 1]]
                        for i in range(len(n_images_in_text))
                    ]
                else:
                    ls_ls_images = [cast(list[Image.Image], images)]
            elif (
                not isinstance(images, (list, tuple))
                and not isinstance(images[0], (list, tuple))
                and not is_image_or_image_url(images[0][0])
            ):
                raise ValueError(
                    "Invalid input images. Please provide a single image or a list of images or a list of list of images."
                )
            else:
                ls_ls_images = cast(list[list[Image.Image]], images)

            n_images_in_images = [len(imgs) for imgs in ls_ls_images]

            image_inputs = self.image_processor(ls_ls_images)
            inputs.update(image_inputs)

            if text is not None:
                if n_images_in_images != n_images_in_text:
                    raise ValueError(
                        "The number of images provided does not match the number of image tokens in the text prompts."
                    )

                image_rows = inputs.pop("rows", [[0] * len(text)])
                image_cols = inputs.pop("cols", [[0] * len(text)])

                image_token = self.image_token
                global_img_token = self.global_image_token

                prompt_strings = []
                batch_image_seq_lengths = []

                for sample, sample_rows, sample_cols in zip(text, image_rows, image_cols):
                    image_prompt_strings = []
                    image_seq_lengths = []

                    for n_rows, n_cols in zip(sample_rows, sample_cols):
                        image_prompt_string = ""
                        count = 0
                        for idx, (n_h, n_w) in enumerate(zip(n_rows, n_cols)):
                            if len(n_rows) > 1:
                                raise NotImplementedError(
                                    "Multiple images per sample are not supported yet.")

                            image_prompt_string += global_img_token
                            count += 1
                            image_prompt_string += image_token * image_seq_len
                            count += image_seq_len

                            if n_h == 1 and n_w == 1:
                                continue

                            for i in range(n_h):
                                for j in range(n_w):
                                    image_prompt_string += getattr(
                                        self.tokenizer, f'r{i+1}c{j+1}')
                                    image_prompt_string += image_token * image_seq_len
                                    count += 1 + image_seq_len

                        image_seq_lengths.append(count)
                        image_prompt_strings.append(image_prompt_string)
                    batch_image_seq_lengths.append(image_seq_lengths)
                    split_image = sample.split(image_token)

                    if len(split_image) == 0:
                        raise ValueError(
                            "The image token should be present in the text.")

                    sample = split_image[0]
                    for i, image_prompt_string in enumerate(image_prompt_strings):
                        sample += str(image_prompt_string) + \
                            str(split_image[i + 1])
                    prompt_strings.append(sample)

                text_inputs = self.tokenizer(
                    prompt_strings
                )

                self._check_special_mm_tokens(
                    prompt_strings,
                    text_inputs,
                    modalities=["image"]
                )

                inputs.update(text_inputs)

        elif text is not None:
            if any(n_images_in_text):
                raise ValueError(
                    f"Found {sum(n_images_in_text)} {self.image_token} tokens in the text but no images were passed."
                )
            text_inputs = self.tokenizer(text=text)
            inputs.update(text_inputs)

        return BatchFeature(data=inputs, tensor_type=None)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):


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

        config: NanoVLMConfig = cast(
            NanoVLMConfig, vllm_config.model_config.hf_config)

        self.config = config

        self.vision_encoder = ViT(
            config, prefix=maybe_prefix(prefix, "vision_encoder"))

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
