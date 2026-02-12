from vllm.model_executor.models.registry import _ModelInfo
from vllm.model_executor.models.nano_vlm import NanoVLMForConditionalGeneration
from vllm.model_executor.models.idefics3 import Idefics3ForConditionalGeneration
from safetensors.torch import load_file, save_file

def load_model_info():

    output = _ModelInfo.from_model_cls(NanoVLMForConditionalGeneration)

    print(output)


def test_load_weight():
    state_dict = load_file("pretrained_models/SmolVLM-135M/model.safetensors", device="cuda")

    print(dict(state_dict).keys())

    # state_dict = load_file("model.safetensors", device="cpu")
    print(state_dict["model.decoder.rotary_embd.inv_freq"])


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

if __name__ == "__main__":
    # test_load_weight()
    # add_prefix_to_safetensors(
    #     input_path="pretrained_models/SmolVLM-135M/ori_model.safetensors",
    #     output_path="pretrained_models/SmolVLM-135M/model.safetensors",
    #     prefix="model."
    # )

    test_load_weight()
