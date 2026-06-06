from .openai_llm_node import LLMImageSelectorNode, OpenAILLMNode
from .beatdrop_selector_node import BeatDropSelectorNode
from .outfit_reference_judge import AlphaRavisOutfitReferenceJudgeNode

NODE_CLASS_MAPPINGS = {
    "OpenAILLMNode": OpenAILLMNode,
    "LLMImageSelectorNode": LLMImageSelectorNode,
    "BeatDropSelectorNode": BeatDropSelectorNode,
    "AlphaRavisOutfitReferenceJudgeNode": AlphaRavisOutfitReferenceJudgeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAILLMNode": "OpenAI Compatible LLM",
    "LLMImageSelectorNode": "LLM Image Selector",
    "BeatDropSelectorNode": "🎵 BeatDrop Selector (Window-Aware)",
    "AlphaRavisOutfitReferenceJudgeNode": "👗 Oufit Reference Judge (Vision LLM)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
