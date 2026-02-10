from transformers.configuration_utils import PretrainedConfig


class NanoVLMConfig(PretrainedConfig):
    model_type = "nano_vlm"

    def __init__(
        self,
        vit_hidden_dim: int = 768,
        vit_inter_dim: int = 4 * 768,
        vit_patch_size: int = 16,
        vit_img_size: int = 512,
        vit_n_heads: int = 12,
        vit_dropout: float = 0.0,
        vit_n_blocks: int = 12,
        vit_ln_eps: float = 1e-6,
        vit_cls_flag: bool = False,
        vit_model_type: str = 'google/siglip2-base-patch16-512',
        lm_hidden_dim: int = 960,
        lm_inter_dim: int = 2560,
        lm_rms_eps: float = 1e-5,
        lm_re_base: int = 100000,
        lm_max_position_embeddings: int = 8192,
        lm_base_vocab_size: int = 49152,
        # Number of extra tokens for the VLM (image start, image end, image token)
        extra_token_amount: int = 66,
        # NOTE: Qwen already includes image tokens but for now will use extra_token_amount to be safe
        lm_vocab_size: int = 49152 + 66,
        lm_n_heads: int = 15,
        lm_n_kv_heads: int = 5,
        lm_dropout: float = 0.0,
        lm_n_blocks: int = 32,
        lm_attn_scaling: float = 1.0,
        lm_max_length: int = 4096,
        # Decide if the LM expects tokens or embeddings as input (if using as a backbone for the VLM, set to False)
        lm_use_tokens: bool = False,
        # Decide if you want to tie the LM Head weight to the token embedding weights
        lm_tie_weights: bool = True,
        # 'HuggingFaceTB/SmolLM2-135M' #
        lm_model_type: str = 'HuggingFaceTB/SmolLM2-135M-Instruct',
        lm_tokenizer: str = 'HuggingFaceTB/SmolLM2-135M-Instruct',
        lm_chat_template: str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",

        mp_pixel_shuffle_factor: int = 4,
        mp_image_token_length: int = 64,
        max_img_size: int = 512,
        resize_to_max_side_len: bool = True,
        vlm_extra_tokens: dict[str, str] = {},
        **kwargs
    ):
        self.vit_hidden_dim = vit_hidden_dim
        self.vit_inter_dim = vit_inter_dim
        self.vit_patch_size = vit_patch_size
        self.vit_img_size = vit_img_size
        self.vit_n_heads = vit_n_heads
        self.vit_dropout = vit_dropout
        self.vit_n_blocks = vit_n_blocks
        self.vit_ln_eps = vit_ln_eps
        self.vit_cls_flag = vit_cls_flag
        self.vit_model_type = vit_model_type

        self.lm_hidden_dim = lm_hidden_dim
        self.lm_inter_dim = lm_inter_dim
        self.lm_rms_eps = lm_rms_eps
        self.lm_re_base = lm_re_base
        self.lm_max_position_embeddings = lm_max_position_embeddings
        self.lm_base_vocab_size = lm_base_vocab_size
        self.extra_token_amount = extra_token_amount
        self.lm_vocab_size = lm_vocab_size
        self.lm_n_heads = lm_n_heads
        self.lm_n_kv_heads = lm_n_kv_heads
        self.lm_dropout = lm_dropout
        self.lm_n_blocks = lm_n_blocks
        self.lm_attn_scaling = lm_attn_scaling
        self.lm_max_length = lm_max_length
        self.lm_use_tokens = lm_use_tokens
        self.lm_tie_weights = lm_tie_weights
        self.lm_model_type = lm_model_type
        self.lm_tokenizer = lm_tokenizer
        self.lm_chat_template = lm_chat_template

        self.mp_pixel_shuffle_factor = mp_pixel_shuffle_factor
        self.mp_image_token_length = mp_image_token_length
        self.max_img_size = max_img_size
        self.resize_to_max_side_len = resize_to_max_side_len

        self.vlm_extra_tokens = vlm_extra_tokens

        # NOTE: Add some additional attributes for vllm integration
        self.num_hidden_layers = lm_n_blocks
        self.num_attention_heads = lm_n_heads
        self.hidden_size = lm_hidden_dim
        self.vocab_size = lm_vocab_size
        self.num_key_value_heads = lm_n_kv_heads


        super().__init__(**kwargs)
