from .openai_llm_node import LLMImageSelectorNode, OpenAILLMNode

NODE_CLASS_MAPPINGS = {
    "OpenAILLMNode": OpenAILLMNode,
    "LLMImageSelectorNode": LLMImageSelectorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAILLMNode": "OpenAI Compatible LLM",
    "LLMImageSelectorNode": "LLM Image Selector",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
