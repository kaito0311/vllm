
import math
from itertools import accumulate
from collections.abc import Mapping, Sequence, Iterable
from typing import Optional, cast, Annotated, Literal, TypeAlias, Union

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from transformers import Idefics3ImageProcessor, ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_base import BatchFeature as ImageBatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.transformers_utils.configs.nano_vlm import NanoVLMConfig
from vllm.model_executor.models.utils import IntermediateTensors, maybe_prefix
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import BaseProcessingInfo, BaseDummyInputsBuilder, BaseMultiModalProcessor, PromptReplacement
from vllm.multimodal.processing import (
    BaseProcessingInfo,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor

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
                processed_images.append(torch.concat(row_images))

        processed_images = torch.stack(processed_images)  # (bn, p, c, h, w)
        rows = rows  # (bn, 1)
        cols = cols  # (bn, 1)

        bn, p, _, h, w = processed_images.shape

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
                        if len(sample_rows) > 1:
                            raise NotImplementedError(
                                "Multiple images per sample are not supported yet.")
                        image_prompt_string = ""
                        count = 0
                        for idx, (n_h, n_w) in enumerate([(n_rows, n_cols)]):
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
        return_tensors = kwargs.get("return_tensors", None)

        # type: ignore
        return BatchFeature(data=inputs, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        ...


class NanoVLMProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> NanoVLMProcessor:
        image_processor = NanoVLMImageProcessor(
            max_image_size=self.ctx.model_config.hf_config.max_img_size,
            patch_size=self.ctx.model_config.hf_config.vit_img_size,
            resize_to_max_size_len=self.ctx.model_config.hf_config.resize_to_max_side_len,
        )

        processor = NanoVLMProcessor(
            image_processor=image_processor,
            tokenizer=self.ctx.tokenizer,
            image_seq_len=self.ctx.model_config.hf_config.mp_image_token_length,
        )

        return processor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}

    def get_num_patches(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: NanoVLMProcessor | None,
    ) -> int:
        grid_w, grid_h = self._get_image_feature_grid_size(
            image_width=image_width,
            image_height=image_height,
            processor=processor,
        )

        return grid_w * grid_h + 1

    def _resize_output_size(
        self,
        *,
        height: int,
        width: int,
        max_len: int | None = None,
        min_len: int = 1,
        max_size: int | None = None,
    ) -> tuple[int, int]:
        # Set default value for max_len if not provided
        max_len = max(height, width) if max_len is None else max_len
        aspect_ratio = width / height

        # Handle the maximum size constraint
        if max_size is not None:
            max_len = min(max_len, max_size)

        # Adjust dimensions according to the aspect ratio
        if width >= height:
            width = max_len
            height = int(width / aspect_ratio)
        else:
            height = max_len
            width = int(height * aspect_ratio)

        # Ensure both width and height are even (if needed)
        height += height % 2
        width += width % 2

        # Ensure dimensions are not smaller than the minimum length
        height = max(height, min_len)
        width = max(width, min_len)

        return height, width

    def _get_resize_output_image_size(
        self,
        *,
        image_width: int,
        image_height: int,
        resolution_max_side: int,
    ) -> tuple[int, int]:
        hf_processor = self.get_hf_processor()
        image_processor: NanoVLMImageProcessor = hf_processor.image_processor
        max_image_size = image_processor.max_image_size
        if resolution_max_side > max_image_size:
            raise ValueError(
                "`resolution_max_side` cannot be larger than `max_image_size`"
            )

        height, width = image_height, image_width

        # Find the output size, when rescaling the longest edge to max_len and
        # preserving the aspect ratio
        height, width = self._resize_output_size(
            height=height, width=width, max_len=resolution_max_side
        )
        return height, width

    def _get_image_feature_grid_size(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: NanoVLMProcessor | None,
    ) -> tuple[int, int]:
        if processor is None:
            processor = self.get_hf_processor()

        image_processor: NanoVLMImageProcessor = processor.image_processor

        max_image_size = image_processor.max_image_size

        assert image_processor.resize_to_max_size_len is True, (
            "The image_processor must have `resize_to_max_size_len=True` to ensure the `size` is equal to `max_image_size`."
        )
        size = image_processor.max_image_size

        assert size % max_image_size == 0, (
            "`longest_edge` in image_processor's `size` must be divisible by "
            "`longest_edge` in `max_image_size`, this may be caused by "
            "incorrect mm_kwargs override."
        )

        resized_height, resized_width = self._get_resize_output_image_size(
            image_width=image_width,
            image_height=image_height,
            resolution_max_side=size,
        )
        if resized_height > max_image_size or resized_width > max_image_size:
            grid_h = math.ceil(resized_height / max_image_size)
            grid_w = math.ceil(resized_width / max_image_size)
        else:
            grid_h = grid_w = 0
        return grid_w, grid_h

    def _get_image_token(
        self, processor: NanoVLMProcessor | None
    ) -> tuple[str, str]:
        if processor is None:
            processor = self.get_hf_processor()

        image_token = processor.image_token
        global_image_token = processor.global_image_token

        return image_token, global_image_token

    def get_image_repl(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: NanoVLMProcessor | None
    ) -> str:
        if processor is None:
            processor = self.get_hf_processor()

        image_token, global_image_token = self._get_image_token(processor)

        image_seq_len = processor.image_seq_len

        grid_placeholders = "<row_{n_h}_col_{n_w}>"

        p_image = image_token * image_seq_len

        global_img_placeholder = global_image_token + p_image
        tile_img_placeholder = grid_placeholders + p_image

        grid_w, grid_h = self._get_image_feature_grid_size(
            image_width=image_width,
            image_height=image_height,
            processor=processor,
        )

        if grid_w == 0 and grid_h == 0:
            return global_img_placeholder

        tiles_placeholder = list[str]()
        for i in range(grid_h):
            for j in range(grid_w):
                tiles_placeholder.append(
                    tile_img_placeholder.format(n_h=i + 1, n_w=j + 1)
                )

        return "".join(
            [
                global_img_placeholder,
                *tiles_placeholder
            ]
        )


class NanoVLMDummyInputsBuilder(BaseDummyInputsBuilder[NanoVLMProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token, _ = self.info._get_image_token(processor)

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        hf_processor = self.info.get_hf_processor()
        image_processor: NanoVLMImageProcessor = hf_processor.image_processor
        longest_edge = image_processor.max_image_size

        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=longest_edge,
                height=longest_edge,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class NanoVLMMultiModalProcessor(BaseMultiModalProcessor[NanoVLMProcessingInfo]):
    # TODO
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Text-only input not supported in composite processor
        if not (images := mm_data.get("images", [])):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        mm_kwargs = {"input_data_format": "channels_last", **mm_kwargs}
        processed_outputs = super()._call_hf_processor(
            prompt,
            mm_data,
            mm_kwargs,
            tok_kwargs,
        )

        mm_items = self.info.parse_mm_data({"image": images}, validate=False)
        parsed_images = mm_items.get_items("image", ImageProcessorItems)
        image_sizes = [
            parsed_images.get_image_size(i) for i in range(len(parsed_images))
        ]
        hf_processor = self.info.get_hf_processor(**mm_kwargs)

        num_patches = [
            self.info.get_num_patches(
                image_width=size.width,
                image_height=size.height,
                processor=hf_processor,
            )
            for size in image_sizes
        ]
        processed_outputs["num_patches"] = torch.tensor(num_patches)

        # Remove the extra batch dimension
        processed_outputs["pixel_values"].squeeze_(0)
        processed_outputs["pixel_attention_mask"].squeeze_(0)

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_patches = hf_inputs.get("num_patches", torch.empty(0))

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", num_patches),
            pixel_attention_mask=MultiModalFieldConfig.flat_from_sizes(
                "image", num_patches
            ),
            image_embeds=MultiModalFieldConfig.batched("image"),
            num_patches=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_token, _ = self.info._get_image_token(hf_processor)

        def get_replacement_nanovlm(item_idx: int) -> PromptUpdateDetails:
            images = mm_items.get_items("image", ImageProcessorItems)

            image_size = images.get_image_size(item_idx)

            image_repl = self.info.get_image_repl(
                image_width=image_size.width,
                image_height=image_size.height,
                processor=hf_processor,
            )

            return PromptUpdateDetails.select_text(
                image_repl,
                embed_text=image_token,
            )

        return [
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=get_replacement_nanovlm,
            )
        ]


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
            expected_h = expected_w = self.config.vit_img_size

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
        return self.decoder(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

from vllm.transformers_utils.configs.nano_vlm import NanoVLMConfig
from vllm.model_executor.models.module_mapping import MultiModelKeys


class SimpleLogitsProcessor(LogitsProcessor):
    def _get_logits(self, hidden_states: torch.Tensor, lm_head: VocabParallelEmbedding, embedding_bias: torch.Tensor | None) -> torch.Tensor | None:
        logits = lm_head(hidden_states)
        if embedding_bias is not None:
            logits += embedding_bias
        
        logits = self._gather_logits(logits)

        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        
        return logits
        

@MULTIMODAL_REGISTRY.register_processor(
    NanoVLMMultiModalProcessor,
    info=NanoVLMProcessingInfo,
    dummy_inputs=NanoVLMDummyInputsBuilder,
)
class NanoVLMForConditionalGeneration(nn.Module, SupportsMultiModal):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: NanoVLMConfig = cast(NanoVLMConfig, vllm_config.model_config.hf_config)
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        with self._mark_composite_model(
            vllm_config,
            language_targets=LanguageModel,
            tower_targets={"image": ViT}
        ):
            self.model = NanoVLMModel(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "model")
            )

        self.image_token_id = self.config.image_token_id
        self.logits_processor = SimpleLogitsProcessor(config.lm_vocab_size)

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
            expected_h = expected_w = self.config.vit_img_size

            return NanoVLMImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                num_patches=num_patches,
                resolve_bindings={"h": expected_h, "w": expected_w},
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
        self,
        image_input: ImageInputs,
    ) -> torch.Tensor | list[torch.Tensor]:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        pixel_values = image_input["pixel_values"]
        pixel_attention_mask = image_input["pixel_attention_mask"]

        image_features = self.model.vision_encoder(pixel_values)
        image_features = self.model.MP(image_features)

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
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.model.decoder(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.model.decoder.head, hidden_states)
        return logits

    def get_num_mm_encoder_tokens(
        self,
        num_image_tokens: int,
    ) -> int:
        hf_config = self.config
        scale_factor = hf_config.scale_factor

        return num_image_tokens * scale_factor**2

    def get_num_mm_connector_tokens(
        self,
        num_vision_tokens: int,
    ) -> int:
        hf_config = self.config
        scale_factor = hf_config.scale_factor

        return num_vision_tokens // scale_factor**2

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        from .utils import AutoWeightsLoader

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="model.decoder",
            connector="model.MP",
            tower_model="model.vision_encoder",
        )