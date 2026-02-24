from vllm import LLM, SamplingParams 
from PIL import Image 
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# vllm serve "./pretrained_models/SmolVLM-135M" --kv-cache-memory-bytes 0 --cpu-offload-gb 0 --max-num-seqs 1 --max-model-len 4096 --port 8080 --no-enable-prefix-caching --dtype float --no-enforce-eager --no-async-scheduling --distributed-executor-backend uni

def test_vllm_chat():

    llm = LLM("./pretrained_models/SmolVLM-135M", enforce_eager=True, kv_cache_memory_bytes=0, cpu_offload_gb=0.0, max_num_seqs=1, max_model_len=8192, enable_prefix_caching=False, dtype="float")
    llm._cached_repr = "<vLLM.LLM Object - Debug Mode>"

    image = Image.open("images/test_image.jpg").convert("RGB")

    conversation = [
        {"role": "assistant", "content": "Paris is capital of"},
    ]

    sampling_params = SamplingParams(top_k=50, top_p=0.9, temperature=0.5)

    # Perform inference and log output.
    outputs = llm.chat(conversation, sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

def test_vllm_generate():

    llm = LLM("./pretrained_models/SmolVLM-135M", enforce_eager=True, kv_cache_memory_bytes=0, cpu_offload_gb=0.0, max_num_seqs=1, max_model_len=8192, enable_prefix_caching=False, dtype="float")
    llm._cached_repr = "<vLLM.LLM Object - Debug Mode>"

    prompt = "Paris is the capital of"

    image = Image.open("images/test_image.jpg").convert("RGB")

    outputs = llm.generate(
        ["Paris is the capital of"],
    ) 

    for o in outputs:
        generated_text = o.outputs[0].text
        print("Generated Text:\n", generated_text)
    

def main():
    print("Testing vLLM chat...")
    test_vllm_chat()
    # test_vllm_generate()


if __name__ == "__main__":
    main()

