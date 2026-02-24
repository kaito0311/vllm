from vllm import LLM, SamplingParams 
from PIL import Image 
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# vllm serve "./pretrained_models/SmolVLM-256M-Instruct" --kv-cache-memory-bytes 0 --cpu-offload-gb 8 --max-num-seqs 1 --max-model-len 4096 --port 8080

def test_vllm_generate():
    # llm = LLM("./pretrained_models/SmolVLM-256M-Instruct", kv_cache_memory_bytes=0, cpu_offload_gb=8, max_num_seqs=1, max_model_len=4096)
    llm = LLM("HuggingFaceTB/SmolVLM-256M-Instruct", kv_cache_memory_bytes=0, cpu_offload_gb=8, max_num_seqs=1, max_model_len=4096)
    # llm._cached_repr = "<vLLM.LLM Object - Debug Mode>"

    prompt = "USER: hello how are you \nASSISTANT:"

    image = Image.open("images/test_image.jpg").convert("RGB")

    outputs = llm.generate(
        ["Paris is the capital of"],
        sampling_params=sampling_params
    ) 

    for o in outputs:
        generated_text = o.outputs[0].text
        print("Generated Text:\n", generated_text)

def test_vllm_chat():

    llm = LLM("HuggingFaceTB/SmolVLM-256M-Instruct")
    llm._cached_repr = "<vLLM.LLM Object - Debug Mode>"

    image = Image.open("images/ava.webp").convert("RGB")

    conversation = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello, What's your name?"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"},
        {
            "role": "user",
            "content": [
           
                {
                    "type": "image_pil",
                    "image_pil": image,
                },
           
                {
                    "type": "text",
                    "text": "What's in these images?",
                },
            ],
        },
    ]

    # Perform inference and log output.
    outputs = llm.chat(conversation)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

def main():

    print("Testing vLLM chat...")
    test_vllm_chat()
    
    # print("Testing vLLM generate...")
    # test_vllm_generate()


if __name__ == "__main__":
    main()

