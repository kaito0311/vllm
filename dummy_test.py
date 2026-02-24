from vllm.model_executor.models.registry import _ModelInfo
from vllm.model_executor.models.nano_vlm import NanoVLMForConditionalGeneration
from vllm.model_executor.models.idefics3 import Idefics3ForConditionalGeneration
from safetensors.torch import load_file, save_file

def load_model_info():

    output = _ModelInfo.from_model_cls(NanoVLMForConditionalGeneration)

    print(output)


def test_load_weight():
    path = "pretrained_models/SmolVLM-256M-Instruct/model.safetensors"
    # path = "/media/minhdt/DATA/Code/test_ocr/training_tiny_llm/nanoVLM_Qwen/checkpoints/pretrained_continue/step_39500/model.safetensors"
    state_dict = load_file(path, device="cuda")

    print(dict(state_dict).keys())

    # state_dict = load_file("model.safetensors", device="cpu")
    # print(state_dict["model.decoder.rotary_embd.inv_freq"])
    # print(state_dict.keys())

def delete_a_key_from_safetensors(input_path: str, output_path: str, key_to_delete: str):
    # Load
    state_dict = load_file(input_path)
    
    # Remove the specified key
    if key_to_delete in state_dict:
        del state_dict[key_to_delete]
        print(f"Deleted key: {key_to_delete}")
    else:
        print(f"Key not found: {key_to_delete}")
    
    # Save the modified state dict
    save_file(state_dict, output_path)
    print(f"Saved → {output_path}")
    print(f"Keys count: {len(state_dict)}")
    print(f"Example key: {next(iter(state_dict))}")


def add_prefix_to_safetensors(
    input_path: str,
    output_path: str,
    prefix: str = "model.",           # ← change this
    strip_old_prefix: str = None     # optional: remove existing prefix first
):
    # Load
    state_dict = load_file(input_path)
    
    new_state_dict = {}
    
    for k, v in state_dict.items():
        new_key = k
        
        # Optional: remove existing prefix first (common when converting)
        if strip_old_prefix and new_key.startswith(strip_old_prefix):
            new_key = new_key[len(strip_old_prefix):]
            
        # Add new prefix
        new_key = prefix + new_key
        
        new_state_dict[new_key] = v
    
    # Save
    save_file(new_state_dict, output_path)
    print(f"Saved → {output_path}")
    print(f"Keys count: {len(state_dict)} → {len(new_state_dict)}")
    print(f"Example new key: {next(iter(new_state_dict))}")

def copy_head_weight_to_token_embed_weight(
    input_path: str,
    output_path: str,
    head_key: str = "model.lm_head.weight",
    embed_key: str = "model.shared.weight"
):
    # Load
    state_dict = load_file(input_path)
    
    # Copy
    state_dict[embed_key] = state_dict[head_key].clone()
    
    # Save
    save_file(state_dict, output_path)
    print(f"Saved → {output_path}")
    print(f"Keys count: {len(state_dict)}")
    print(f"Example key: {next(iter(state_dict))}")

if __name__ == "__main__":
    # test_load_weight()
    # add_prefix_to_safetensors(
    #     input_path="pretrained_models/SmolVLM-135M/ori_model.safetensors",
    #     output_path="pretrained_models/SmolVLM-135M/model.safetensors",
    #     prefix="model."
    # )

    test_load_weight()
    # delete_a_key_from_safetensors(
    #     input_path="pretrained_models/SmolVLM-135M/model.safetensors",
    #     output_path="pretrained_models/SmolVLM-135M/model_no_inv_freq.safetensors",
    #     key_to_delete="model.decoder.rotary_embd.inv_freq"
    # )

    # copy_head_weight_to_token_embed_weight(
    #     input_path="pretrained_models/SmolVLM-135M/model.safetensors",
    #     output_path="pretrained_models/SmolVLM-135M/model_with_shared_embed.safetensors",
    #     head_key="model.decoder.head.weight",
    #     embed_key="model.decoder.token_embedding.weight"
    # )