# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import copy
from pathlib import Path
from typing import TypeAlias, Any
import json 

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.transformers_utils.config import get_sentence_transformer_tokenizer_config

from .protocol import TokenizerLike


NanoVLMTokenizer: TypeAlias = PreTrainedTokenizer | PreTrainedTokenizerFast

def get_cached_tokenizer(tokenizer: NanoVLMTokenizer) -> NanoVLMTokenizer:
    """
    By default, transformers will recompute multiple tokenizer properties
    each time they are called, leading to a significant slowdown.
    This proxy caches these properties for faster access.
    """
    cached_tokenizer = copy.copy(tokenizer)

    tokenizer_all_special_ids = tokenizer.all_special_ids
    tokenizer_all_special_tokens = tokenizer.all_special_tokens
    tokenizer_vocab = tokenizer.get_vocab()
    tokenizer_len = len(tokenizer)

    max_token_id = max(tokenizer_vocab.values())
    # Some tokenizers (e.g., QwenTokenizer) have special tokens that
    # are added and included in the implementation of the vocab_size
    # property, but not in get_vocab(); if there is an implementation
    # of vocab size, we should take the greater value.
    if hasattr(tokenizer, "vocab_size"):
        with contextlib.suppress(NotImplementedError):
            max_token_id = max(max_token_id, tokenizer.vocab_size)

    class CachedTokenizer(tokenizer.__class__):  # type: ignore
        @property
        def all_special_ids(self) -> list[int]:
            return tokenizer_all_special_ids

        @property
        def all_special_tokens(self) -> list[str]:
            return tokenizer_all_special_tokens

        @property
        def max_token_id(self) -> int:
            return max_token_id

        def get_vocab(self) -> dict[str, int]:
            return tokenizer_vocab

        def __len__(self) -> int:
            return tokenizer_len

        def __reduce__(self):
            return get_cached_tokenizer, (tokenizer,)

    CachedTokenizer.__name__ = f"Cached{tokenizer.__class__.__name__}"

    cached_tokenizer.__class__ = CachedTokenizer
    return cached_tokenizer

def get_nanovlm_tokenizer(
        lm_tokenizer_name: str,
        *args,
        extra_special_tokens: list[str] | None = None,
        chat_template: str | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
        
    ): 

    assert lm_tokenizer_name is not None, "lm_tokenizer_name must be provided for NanoVLM tokenizer."

    tokenizer_init_kwargs : dict[str, Any] = {"use_fast": True}

    if extra_special_tokens is not None:
        tokenizer_init_kwargs["extra_special_tokens"] = extra_special_tokens
    
    if chat_template is not None:
        tokenizer_init_kwargs["chat_template"] = chat_template

    tokenizer = AutoTokenizer.from_pretrained(
        lm_tokenizer_name,
        *args,
        trust_remote_code=trust_remote_code,
        revision=revision,
        cache_dir=download_dir,
        **tokenizer_init_kwargs,
        **kwargs
    )

    return tokenizer
    

class CachedNanoVLMTokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args, 
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> NanoVLMTokenizer:
        try:
            if (local_path := Path(path_or_repo_id)).exists():
                # load config.json file 
                config_path = local_path / "config.json"
                if config_path.exists():
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    # You can now use the config dictionary as needed
                tokenizer = get_nanovlm_tokenizer(
                    lm_tokenizer_name=config.get("lm_tokenizer", None),
                    extra_special_tokens=config.get("vlm_extra_tokens", None),
                    chat_template=config.get("lm_chat_template", None),
                    *args,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    download_dir=download_dir,
                    **kwargs,
                )

                return get_cached_tokenizer(tokenizer)
            else:
                # TODO
                raise NotImplementedError("Loading NanoVLM tokenizer config from remote repositories is not implemented yet.")
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer config from {path_or_repo_id}: {e}")