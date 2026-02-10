from vllm.model_executor.models.registry import _ModelInfo
from vllm.model_executor.models.nano_vlm import NanoVLMForConditionalGeneration
from vllm.model_executor.models.idefics3 import Idefics3ForConditionalGeneration


output = _ModelInfo.from_model_cls(NanoVLMForConditionalGeneration)

print(output)