from vllm import LLM, SamplingParams 
from PIL import Image 
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    # llm = LLM("./pretrained_models/SmolVLM-256M-Instruct", kv_cache_memory_bytes=0, cpu_offload_gb=8, max_num_seqs=1, max_model_len=4096)
    llm = LLM("HuggingFaceTB/SmolVLM-256M-Instruct", kv_cache_memory_bytes=0, cpu_offload_gb=8, max_num_seqs=1, max_model_len=4096)
    llm._cached_repr = "<vLLM.LLM Object - Debug Mode>"

    prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

    image = Image.open("images/test_image.jpg").convert("RGB")

    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            }
        }
    ) 

    for o in outputs:
        generated_text = o.outputs[0].text
        print("Generated Text:\n", generated_text)
    


if __name__ == "__main__":
    main()

