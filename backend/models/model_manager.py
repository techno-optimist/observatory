"""
Model Manager

Handles loading, caching, and unloading of various model types:
- Sentence Transformers (fastest, good for quick exploration)
- HuggingFace Transformers (full access to any layer)
- GGUF models via llama-cpp (for quantized models on CPU)

Includes model provenance tracking for reproducibility.
"""

import torch
import hashlib
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    SENTENCE_TRANSFORMER = "sentence-transformer"
    HUGGINGFACE = "huggingface"
    GGUF = "gguf"


@dataclass
class ModelProvenance:
    """
    Tracks model provenance for reproducibility.

    Stores the exact model version, weight hash, and load time
    to ensure experiments can be reproduced precisely.
    """
    model_id: str
    revision: Optional[str] = None  # Git commit hash from HuggingFace
    sha256: Optional[str] = None    # Hash of model weights (first layer)
    load_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    model_type: str = "unknown"
    source: str = "huggingface"     # huggingface, local, gguf

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "sha256": self.sha256,
            "load_timestamp": self.load_timestamp,
            "model_type": self.model_type,
            "source": self.source
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelProvenance":
        return cls(
            model_id=data["model_id"],
            revision=data.get("revision"),
            sha256=data.get("sha256"),
            load_timestamp=data.get("load_timestamp", datetime.utcnow().isoformat()),
            model_type=data.get("model_type", "unknown"),
            source=data.get("source", "huggingface")
        )


@dataclass
class ModelInfo:
    model_id: str
    model_type: ModelType
    embedding_dim: int
    num_layers: int
    device: str
    quantization: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "device": self.device,
            "quantization": self.quantization
        }


class ModelManager:
    """Manages model lifecycle - loading, caching, unloading."""

    # Popular models for quick selection
    RECOMMENDED_MODELS = {
        "sentence-transformer": [
            ("all-MiniLM-L6-v2", "Fast, 384d, good general purpose"),
            ("all-mpnet-base-v2", "Higher quality, 768d"),
            ("paraphrase-multilingual-MiniLM-L12-v2", "Multilingual, 384d"),
        ],
        "huggingface": [
            ("microsoft/phi-2", "2.7B, excellent for size"),
            ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "1.1B, fast"),
            ("mistralai/Mistral-7B-v0.1", "7B, high quality"),
            ("meta-llama/Llama-2-7b-hf", "7B, requires auth"),
        ]
    }

    def __init__(self, device: Optional[str] = None, max_cached_models: int = 2):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_cached_models = max_cached_models
        self._cache: Dict[str, Tuple[Any, Any, ModelInfo]] = {}  # model_id -> (model, tokenizer, info)
        self._load_order: List[str] = []  # For LRU eviction
        self._provenance: Dict[str, ModelProvenance] = {}  # model_id -> provenance

        logger.info(f"ModelManager initialized. Device: {self.device}")

    def get_available_models(self) -> Dict[str, List[Tuple[str, str]]]:
        """Return list of recommended models by type."""
        return self.RECOMMENDED_MODELS

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded."""
        return model_id in self._cache

    def get_loaded_models(self) -> List[ModelInfo]:
        """Get info about all currently loaded models."""
        return [info for _, _, info in self._cache.values()]

    def load_model(
        self,
        model_id: str,
        model_type: ModelType = ModelType.SENTENCE_TRANSFORMER,
        force_reload: bool = False
    ) -> ModelInfo:
        """
        Load a model into memory.

        Args:
            model_id: HuggingFace model ID or path
            model_type: Type of model to load
            force_reload: If True, reload even if cached
        """
        if model_id in self._cache and not force_reload:
            # Move to end of load order (LRU)
            self._load_order.remove(model_id)
            self._load_order.append(model_id)
            return self._cache[model_id][2]

        # Evict if at capacity
        while len(self._cache) >= self.max_cached_models:
            self._evict_oldest()

        logger.info(f"Loading model: {model_id} (type: {model_type})")

        if model_type == ModelType.SENTENCE_TRANSFORMER:
            model, tokenizer, info, provenance = self._load_sentence_transformer(model_id)
        elif model_type == ModelType.HUGGINGFACE:
            model, tokenizer, info, provenance = self._load_huggingface(model_id)
        elif model_type == ModelType.GGUF:
            model, tokenizer, info, provenance = self._load_gguf(model_id)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self._cache[model_id] = (model, tokenizer, info)
        self._load_order.append(model_id)
        self._provenance[model_id] = provenance

        logger.info(f"Model loaded: {model_id} ({info.embedding_dim}d, {info.num_layers} layers)")
        logger.info(f"Model provenance: revision={provenance.revision}, sha256={provenance.sha256[:16] if provenance.sha256 else 'N/A'}...")
        return info

    def _load_sentence_transformer(self, model_id: str) -> Tuple[Any, Any, ModelInfo, ModelProvenance]:
        """Load a sentence-transformers model."""
        from sentence_transformers import SentenceTransformer

        # Handle short names
        original_id = model_id
        if "/" not in model_id:
            model_id = f"sentence-transformers/{model_id}"

        model = SentenceTransformer(model_id, device=self.device)

        # Get embedding dimension
        embedding_dim = model.get_sentence_embedding_dimension()

        # Sentence transformers don't expose layers directly, use 1
        num_layers = 1

        info = ModelInfo(
            model_id=model_id,
            model_type=ModelType.SENTENCE_TRANSFORMER,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            device=self.device
        )

        # Extract provenance information
        revision = self._get_hf_revision(model_id)
        sha256 = self._compute_model_hash(model)

        provenance = ModelProvenance(
            model_id=model_id,
            revision=revision,
            sha256=sha256,
            model_type=ModelType.SENTENCE_TRANSFORMER.value,
            source="huggingface"
        )

        return model, None, info, provenance

    def _load_huggingface(self, model_id: str) -> Tuple[Any, Any, ModelInfo, ModelProvenance]:
        """Load a HuggingFace transformers model with full layer access."""
        from transformers import AutoModel, AutoTokenizer, AutoConfig

        # Load config first to get architecture info
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with output_hidden_states enabled
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            output_hidden_states=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        model = model.to(self.device)
        model.eval()

        # Get embedding dimension and layer count
        embedding_dim = config.hidden_size
        num_layers = getattr(config, 'num_hidden_layers', 12)

        info = ModelInfo(
            model_id=model_id,
            model_type=ModelType.HUGGINGFACE,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            device=self.device
        )

        # Extract provenance information
        revision = self._get_hf_revision(model_id)
        sha256 = self._compute_model_hash(model)

        provenance = ModelProvenance(
            model_id=model_id,
            revision=revision,
            sha256=sha256,
            model_type=ModelType.HUGGINGFACE.value,
            source="huggingface"
        )

        return model, tokenizer, info, provenance

    def _load_gguf(self, model_id: str) -> Tuple[Any, Any, ModelInfo, ModelProvenance]:
        """Load a GGUF quantized model via llama-cpp-python."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with: "
                "pip install llama-cpp-python"
            )

        # model_id should be a path to .gguf file
        model = Llama(
            model_path=model_id,
            embedding=True,
            n_ctx=2048,
            n_gpu_layers=-1 if self.device == "cuda" else 0
        )

        # Get embedding dimension from model
        embedding_dim = model.n_embd()
        num_layers = model.n_layer() if hasattr(model, 'n_layer') else 1

        info = ModelInfo(
            model_id=model_id,
            model_type=ModelType.GGUF,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            device=self.device,
            quantization="GGUF"
        )

        # For GGUF, compute hash of the file itself
        sha256 = self._compute_file_hash(model_id)

        provenance = ModelProvenance(
            model_id=model_id,
            revision=None,  # GGUF files don't have HF revision
            sha256=sha256,
            model_type=ModelType.GGUF.value,
            source="gguf"
        )

        return model, None, info, provenance

    def _get_hf_revision(self, model_id: str) -> Optional[str]:
        """Get the git revision hash from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download, HfApi
            api = HfApi()
            model_info = api.model_info(model_id)
            return model_info.sha
        except Exception as e:
            logger.warning(f"Could not fetch HF revision for {model_id}: {e}")
            return None

    def _compute_model_hash(self, model: Any) -> Optional[str]:
        """
        Compute a hash of the model weights for verification.

        Uses the first layer's weights to create a fingerprint without
        iterating through the entire model (which could be slow for large models).
        """
        try:
            # Get first parameter tensor
            params = list(model.parameters())
            if not params:
                return None

            # Hash first parameter (usually embedding or first layer)
            first_param = params[0].detach().cpu().numpy().tobytes()
            return hashlib.sha256(first_param).hexdigest()
        except Exception as e:
            logger.warning(f"Could not compute model hash: {e}")
            return None

    def _compute_file_hash(self, file_path: str) -> Optional[str]:
        """Compute SHA256 hash of a file (for GGUF models)."""
        try:
            from pathlib import Path
            path = Path(file_path)
            if not path.exists():
                return None

            # For large files, hash first 10MB only for speed
            hasher = hashlib.sha256()
            with open(path, 'rb') as f:
                # Read first 10MB
                chunk = f.read(10 * 1024 * 1024)
                hasher.update(chunk)

            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"Could not compute file hash: {e}")
            return None

    def get_model(self, model_id: str) -> Tuple[Any, Any, ModelInfo]:
        """Get a loaded model, its tokenizer, and info."""
        if model_id not in self._cache:
            raise ValueError(f"Model not loaded: {model_id}. Call load_model first.")
        return self._cache[model_id]

    def get_provenance(self, model_id: str) -> Optional[ModelProvenance]:
        """Get provenance information for a loaded model."""
        return self._provenance.get(model_id)

    def get_all_provenance(self) -> Dict[str, ModelProvenance]:
        """Get provenance for all loaded models."""
        return dict(self._provenance)

    def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory."""
        if model_id not in self._cache:
            return False

        model, tokenizer, info = self._cache.pop(model_id)
        self._load_order.remove(model_id)

        # Remove provenance (keep for audit trail if desired)
        if model_id in self._provenance:
            del self._provenance[model_id]

        # Clear from GPU if applicable
        del model
        del tokenizer
        if self.device == "cuda":
            torch.cuda.empty_cache()

        logger.info(f"Unloaded model: {model_id}")
        return True

    def _evict_oldest(self):
        """Remove the least recently used model."""
        if not self._load_order:
            return

        oldest = self._load_order[0]
        logger.info(f"Evicting model (LRU): {oldest}")
        self.unload_model(oldest)

    def clear_cache(self):
        """Unload all models."""
        model_ids = list(self._cache.keys())
        for model_id in model_ids:
            self.unload_model(model_id)


# Singleton instance
_manager_instance: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global ModelManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ModelManager()
    return _manager_instance
