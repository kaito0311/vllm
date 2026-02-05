from vllm import LLM, SamplingParams 

long_prefix = "Hello, this is a long prefix. "

prompts = [
    "Hello, how are you?",
    "Tell me a joke. How are you?",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    llm = LLM("HuggingFaceTB/SmolLM2-135M-Instruct", kv_cache_memory_bytes=0, cpu_offload_gb=8, max_num_seqs=1, max_model_len=256)
    llm._cached_repr = "<vLLM.LLM Object - Debug Mode>"


    import time 

    start_time = time.time()
    outputs = llm.generate(
        long_prefix + prompts[0],
        sampling_params=sampling_params
    )
    print("Generation time for single prompt:", time.time() - start_time)

    print(outputs)

    start_time = time.time()
    outputs = llm.generate(
        long_prefix + prompts[1],
        sampling_params=sampling_params
    )
    print(outputs)
    print("Generation time for single prompt:", time.time() - start_time)


if __name__ == "__main__":
    main()

