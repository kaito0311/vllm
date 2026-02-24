# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
import itertools
from collections import defaultdict, deque
from collections.abc import Set
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

import jinja2
import jinja2.ext
import jinja2.meta
import jinja2.nodes
import jinja2.parser
import jinja2.sandbox

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormat,
    ChatTemplateContentFormatOption,
    ChatTemplateResolutionError,
    ConversationMessage,
    load_chat_template,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.inputs import EmbedsPrompt, TextPrompt, TokensPrompt
from vllm.logger import init_logger
from vllm.tokenizers import cached_get_tokenizer
from vllm.tokenizers.nano_vlm import CachedNanoVLMTokenizer, NanoVLMTokenizer
from vllm.transformers_utils.chat_templates import get_chat_template_fallback_path
from vllm.transformers_utils.processor import cached_get_processor
from vllm.utils.func_utils import supports_kw
from .hf import _detect_content_format

from .params import ChatParams
from .protocol import BaseRenderer

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalDataDict, MultiModalUUIDDict
else:
    MultiModalDataDict = dict[str, Any]
    MultiModalUUIDDict = dict[str, Any]


logger = init_logger(__name__)


@lru_cache
def _log_chat_template_content_format(
    chat_template: str | None,  # For caching purposes
    given_format: ChatTemplateContentFormatOption,
    detected_format: ChatTemplateContentFormatOption,
):
    logger.info(
        "Detected the chat template content format to be '%s'. "
        "You can set `--chat-template-content-format` to override this.",
        detected_format,
    )

    if given_format != "auto" and given_format != detected_format:
        logger.warning(
            "You specified `--chat-template-content-format %s` "
            "which is different from the detected format '%s'. "
            "If our automatic detection is incorrect, please consider "
            "opening a GitHub issue so that we can improve it: "
            "https://github.com/vllm-project/vllm/issues/new/choose",
            given_format,
            detected_format,
        )

def resolve_chat_template(
    tokenizer: NanoVLMTokenizer,
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
    *,
    model_config: "ModelConfig",
) -> str | None:

    # 3rd priority: AutoTokenizer chat template
    try:
        return tokenizer.get_chat_template(chat_template, tools=tools)
    except Exception:
        logger.debug(
            "Failed to load AutoTokenizer chat template for %s",
            tokenizer.name_or_path,
            exc_info=True,
        )

    # 4th priority: Predefined fallbacks
    path = get_chat_template_fallback_path(
        model_type=model_config.hf_config.model_type,
        tokenizer_name_or_path=tokenizer.name_or_path,
    )
    if path is not None:
        logger.info_once(
            "Loading chat template fallback for %s as there isn't one "
            "defined on HF Hub.",
            tokenizer.name_or_path,
        )
        chat_template = load_chat_template(path)
    else:
        logger.debug_once(
            "There is no chat template fallback for %s", tokenizer.name_or_path
        )

    return chat_template

def _resolve_chat_template_content_format(
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
    tokenizer: NanoVLMTokenizer,
    *,
    model_config: "ModelConfig",
) -> ChatTemplateContentFormat:
    resolved_chat_template = resolve_chat_template(
        tokenizer,
        chat_template=chat_template,
        tools=tools,
        model_config=model_config,
    )

    jinja_text = (
        resolved_chat_template
        if isinstance(resolved_chat_template, str)
        else load_chat_template(chat_template, is_literal=True)
    )

    detected_format = (
        "string"
        if jinja_text is None
        else _detect_content_format(jinja_text, default="string")
    )

    return detected_format


def resolve_chat_template_content_format(
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
    given_format: ChatTemplateContentFormatOption,
    tokenizer: NanoVLMTokenizer,
    *,
    model_config: "ModelConfig",
) -> ChatTemplateContentFormat:
    if given_format != "auto":
        return given_format

    detected_format = _resolve_chat_template_content_format(
        chat_template,
        tools,
        tokenizer,
        model_config=model_config,
    )

    _log_chat_template_content_format(
        chat_template,
        given_format=given_format,
        detected_format=detected_format,
    )

    return detected_format


def safe_apply_chat_template(
    model_config: "ModelConfig",
    tokenizer: NanoVLMTokenizer,
    conversation: list[ConversationMessage],
    *,
    tools: list[dict[str, Any]] | None = None,
    chat_template: str | None = None,
    tokenize: bool = True,
    **kwargs,
) -> str | list[int]:
    chat_template = resolve_chat_template(
        tokenizer,
        chat_template=chat_template,
        tools=tools,
        model_config=model_config,
    )
    if chat_template is None:
        raise ChatTemplateResolutionError(
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."
        )

    # NOTE: Currently, I don't know what it is, so, I will ignore it. :)
    # resolved_kwargs = resolve_chat_template_kwargs(
    #     tokenizer=tokenizer,
    #     chat_template=chat_template,
    #     chat_template_kwargs=kwargs,
    # )

    try:
        return tokenizer.apply_chat_template(
            conversation=conversation,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            chat_template=chat_template,
            tokenize=tokenize,
            **kwargs
            # **resolved_kwargs,
        )
    # External library exceptions can sometimes occur despite the framework's
    # internal exception management capabilities.
    except Exception as e:
        # Log and report any library-related exceptions for further
        # investigation.
        logger.exception(
            "An error occurred in `transformers` while applying chat template"
        )
        raise ValueError(str(e)) from e

class NanoVLMRenderer(BaseRenderer):
    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        tokenizer_kwargs: dict[str, Any],
    ) -> "BaseRenderer":
        return cls(config, tokenizer_kwargs)

    def __init__(
        self,
        config: ModelConfig,
        tokenizer_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(config)

        self.use_unified_vision_chunk = getattr(
            config.hf_config, "use_unified_vision_chunk", False
        )

        if config.skip_tokenizer_init:
            tokenizer = None
        else:
            tokenizer = cast(
                NanoVLMTokenizer,
                cached_get_tokenizer(
                    tokenizer_cls=CachedNanoVLMTokenizer,  # type: ignore[type-abstract]
                    **tokenizer_kwargs,
                ),
            )

        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> NanoVLMTokenizer | None:
        return self._tokenizer

    def get_tokenizer(self) -> NanoVLMTokenizer:
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer not available when `skip_tokenizer_init=True`")

        return tokenizer

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], TextPrompt | TokensPrompt | EmbedsPrompt]:
        model_config = self.config
        tokenizer = self.get_tokenizer()

        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            model_config,
            content_format="openai", # NOTE: Currently, I will hardcode it to "openai" format as I don't know how to trigger the other formats.
        )

        prompt_raw = safe_apply_chat_template(
            model_config,
            tokenizer,
            conversation,
            **params.get_apply_chat_template_kwargs(),
        )

        # NOTE: use_unified_vision_chunk is currently specific to Kimi-K2.5
        # model which uses unified vision chunks for both images and videos.
        if (
            self.use_unified_vision_chunk
            and mm_uuids is not None
            and mm_data is not None
        ):
            raise NotImplementedError("Synchronous version is not implemented yet.")
            mm_uuids = rebuild_mm_uuids_from_mm_data(mm_uuids, mm_data)

            # get video placeholder, replace it with runtime video-chunk prompts
            video_placeholder = getattr(
                model_config.hf_config, "video_placeholder", None
            )
            prompt_raw = replace_vision_chunk_video_placeholder(
                prompt_raw,
                mm_data,
                video_placeholder,
            )

        prompt = self.render_completion(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], TextPrompt | TokensPrompt | EmbedsPrompt]:
        model_config = self.config
        tokenizer = self.get_tokenizer()

        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            model_config,
            content_format=resolve_chat_template_content_format(
                chat_template=params.chat_template,
                tools=params.chat_template_kwargs.get("tools"),
                given_format=params.chat_template_content_format,
                tokenizer=tokenizer,
                model_config=model_config,
            ),
        )

        prompt_raw = safe_apply_chat_template(
            model_config,
            tokenizer,
            conversation,
            **params.get_apply_chat_template_kwargs(),
        )

        # NOTE: use_unified_vision_chunk is currently specific to Kimi-K2.5
        # model which uses unified vision chunks for both images and videos.
        if (
            self.use_unified_vision_chunk
            and mm_uuids is not None
            and mm_data is not None
        ):
            raise NotImplementedError("Asynchronous version is not implemented yet.")
            # get video placeholder, replace it with runtime video-chunk prompts
            video_placeholder = getattr(
                model_config.hf_config, "video_placeholder", None
            )
            prompt_raw = replace_vision_chunk_video_placeholder(
                prompt_raw,
                mm_data,
                video_placeholder,
            )

        prompt = self.render_completion(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt