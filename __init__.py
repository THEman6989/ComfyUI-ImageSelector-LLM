from .openai_llm_node import LLMImageSelectorNode, OpenAILLMNode
from .beatdrop_selector_node import BeatDropSelectorNode
from .beatdrop_selector_embedding import BeatDropSelectorEmbeddingNode
from .beatdrop_plan_writer import BeatDropPlanWriterPipe
from .beatdrop_config_pipe import BeatDropConfigPipe
from .qwen_vl_embedding_loader import QwenVLEmbeddingLoader, QwenVLEmbeddingUnloader
from .qwen_vl_reranker_loader import QwenVLRerankerLoader, QwenVLRerankerUnloader
from .outfit_reference_judge import AlphaRavisOutfitReferenceJudgeNode

NODE_CLASS_MAPPINGS = {
    "OpenAILLMNode": OpenAILLMNode,
    "LLMImageSelectorNode": LLMImageSelectorNode,
    "BeatDropSelectorNode": BeatDropSelectorNode,
    "BeatDropSelectorEmbeddingNode": BeatDropSelectorEmbeddingNode,
    "BeatDropPlanWriterPipe": BeatDropPlanWriterPipe,
    "BeatDropConfigPipe": BeatDropConfigPipe,
    "QwenVLEmbeddingLoader": QwenVLEmbeddingLoader,
    "QwenVLEmbeddingUnloader": QwenVLEmbeddingUnloader,
    "QwenVLRerankerLoader": QwenVLRerankerLoader,
    "QwenVLRerankerUnloader": QwenVLRerankerUnloader,
    "AlphaRavisOutfitReferenceJudgeNode": AlphaRavisOutfitReferenceJudgeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAILLMNode": "OpenAI Compatible LLM",
    "LLMImageSelectorNode": "LLM Image Selector",
    "BeatDropSelectorNode": "🎵 BeatDrop Selector (Window-Aware)",
    "BeatDropSelectorEmbeddingNode": "🎵 BeatDrop Selector (Embedding)",
    "BeatDropPlanWriterPipe": "🔗 BeatDrop Plan Pipe (PlanWriter)",
    "BeatDropConfigPipe": "🔧 BeatDrop Config Pipe (AI Stack)",
    "QwenVLEmbeddingLoader": "🧠 Qwen VL Embedding Loader",
    "QwenVLEmbeddingUnloader": "🗑 Qwen VL Embedding Unloader",
    "QwenVLRerankerLoader": "🧠 Qwen VL Reranker Loader",
    "QwenVLRerankerUnloader": "🗑 Qwen VL Reranker Unloader",
    "AlphaRavisOutfitReferenceJudgeNode": "👗 Oufit Reference Judge (Vision LLM)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
