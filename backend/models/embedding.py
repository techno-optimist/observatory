"""
Embedding Extractor

Extracts embeddings from various model types with support for:
- Layer selection (extract from any intermediate layer)
- Multiple pooling strategies (mean, CLS, last token, max)
- Batch processing for efficiency
"""

import torch
import numpy as np
from typing import List, Optional, Union, Literal
from dataclasses import dataclass
from enum import Enum
import logging

from .model_manager import ModelManager, ModelType, get_model_manager

logger = logging.getLogger(__name__)


class PoolingStrategy(str, Enum):
    MEAN = "mean"          # Average all token embeddings
    CLS = "cls"            # Use [CLS] token (first token)
    LAST = "last"          # Use last token
    MAX = "max"            # Max pooling across tokens
    WEIGHTED_MEAN = "weighted_mean"  # Weight by attention (if available)


@dataclass
class EmbeddingResult:
    """Result of embedding extraction."""
    embedding: np.ndarray
    layer: int
    pooling: str
    model_id: str
    text: str

    def to_dict(self) -> dict:
        return {
            "embedding": self.embedding.tolist(),
            "layer": self.layer,
            "pooling": self.pooling,
            "model_id": self.model_id,
            "text": self.text,
            "dim": len(self.embedding)
        }


class EmbeddingExtractor:
    """
    Extracts embeddings from loaded models.

    Supports extracting from specific layers and various pooling strategies.
    This is the core component for "lensing" into model latent space.
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or get_model_manager()

    def extract(
        self,
        text: Union[str, List[str]],
        model_id: str,
        layer: int = -1,
        pooling: PoolingStrategy = PoolingStrategy.MEAN,
        normalize: bool = True
    ) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """
        Extract embeddings from text.

        Args:
            text: Single string or list of strings
            model_id: ID of the loaded model
            layer: Which layer to extract from (-1 = last, 0 = embedding layer)
            pooling: How to pool token embeddings
            normalize: Whether to L2-normalize the output

        Returns:
            EmbeddingResult or list of EmbeddingResult
        """
        single_input = isinstance(text, str)
        texts = [text] if single_input else text

        model, tokenizer, info = self.model_manager.get_model(model_id)

        if info.model_type == ModelType.SENTENCE_TRANSFORMER:
            embeddings = self._extract_sentence_transformer(model, texts, normalize)
            # Sentence transformers don't have layer access, use 0
            actual_layer = 0
        elif info.model_type == ModelType.HUGGINGFACE:
            embeddings, actual_layer = self._extract_huggingface(
                model, tokenizer, texts, layer, pooling, normalize, info.num_layers
            )
        elif info.model_type == ModelType.GGUF:
            embeddings = self._extract_gguf(model, texts, normalize)
            actual_layer = 0
        else:
            raise ValueError(f"Unsupported model type: {info.model_type}")

        results = [
            EmbeddingResult(
                embedding=emb,
                layer=actual_layer,
                pooling=pooling.value,
                model_id=model_id,
                text=t
            )
            for t, emb in zip(texts, embeddings)
        ]

        return results[0] if single_input else results

    def _extract_sentence_transformer(
        self,
        model,
        texts: List[str],
        normalize: bool
    ) -> List[np.ndarray]:
        """Extract from sentence-transformers model."""
        embeddings = model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        return [embeddings[i] for i in range(len(texts))]

    def _extract_huggingface(
        self,
        model,
        tokenizer,
        texts: List[str],
        layer: int,
        pooling: PoolingStrategy,
        normalize: bool,
        num_layers: int
    ) -> tuple[List[np.ndarray], int]:
        """
        Extract from HuggingFace model with layer selection.

        This is where the real "lensing" happens - we can look at
        any intermediate representation the model builds.
        """
        # Tokenize
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # Get hidden states from specified layer
        if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
            raise ValueError(
                "Model doesn't return hidden states. "
                "Load with output_hidden_states=True"
            )

        hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden)

        # Normalize layer index
        actual_layer = layer if layer >= 0 else num_layers + layer + 1
        actual_layer = max(0, min(actual_layer, len(hidden_states) - 1))

        layer_output = hidden_states[actual_layer]  # (batch, seq, hidden)
        attention_mask = inputs.get('attention_mask')

        # Apply pooling
        embeddings = self._pool(layer_output, attention_mask, pooling)

        # Normalize if requested
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        # Convert to numpy
        embeddings_np = embeddings.cpu().numpy()

        return [embeddings_np[i] for i in range(len(texts))], actual_layer

    def _pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        strategy: PoolingStrategy
    ) -> torch.Tensor:
        """Apply pooling strategy to hidden states."""

        if strategy == PoolingStrategy.CLS:
            # First token
            return hidden_states[:, 0, :]

        elif strategy == PoolingStrategy.LAST:
            # Last non-padding token
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                return hidden_states[batch_indices, seq_lengths, :]
            else:
                return hidden_states[:, -1, :]

        elif strategy == PoolingStrategy.MAX:
            # Max pooling
            if attention_mask is not None:
                # Mask padding tokens with -inf
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                hidden_states = hidden_states.masked_fill(mask_expanded == 0, float('-inf'))
            return hidden_states.max(dim=1).values

        elif strategy == PoolingStrategy.MEAN:
            # Mean pooling (most common)
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return hidden_states.mean(dim=1)

        elif strategy == PoolingStrategy.WEIGHTED_MEAN:
            # Weight by position (later = more important, roughly mimics attention)
            seq_len = hidden_states.size(1)
            weights = torch.arange(1, seq_len + 1, device=hidden_states.device).float()
            weights = weights / weights.sum()

            if attention_mask is not None:
                weights = weights.unsqueeze(0) * attention_mask
                weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)

            weights = weights.unsqueeze(-1)
            return (hidden_states * weights).sum(dim=1)

        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")

    def _extract_gguf(
        self,
        model,
        texts: List[str],
        normalize: bool
    ) -> List[np.ndarray]:
        """Extract embeddings from GGUF model."""
        embeddings = []
        for text in texts:
            emb = model.embed(text)
            emb = np.array(emb)
            if normalize:
                emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        return embeddings

    def extract_multi_layer(
        self,
        text: str,
        model_id: str,
        layers: Optional[List[int]] = None,
        pooling: PoolingStrategy = PoolingStrategy.MEAN
    ) -> dict[int, np.ndarray]:
        """
        Extract embeddings from multiple layers at once.

        Useful for analyzing how representations evolve through the model.
        """
        model, tokenizer, info = self.model_manager.get_model(model_id)

        if info.model_type != ModelType.HUGGINGFACE:
            raise ValueError("Multi-layer extraction only supported for HuggingFace models")

        # Default to all layers
        if layers is None:
            layers = list(range(info.num_layers + 1))  # +1 for embedding layer

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states
        attention_mask = inputs.get('attention_mask')

        result = {}
        for layer in layers:
            if 0 <= layer < len(hidden_states):
                layer_output = hidden_states[layer]
                pooled = self._pool(layer_output, attention_mask, pooling)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
                result[layer] = pooled.cpu().numpy()[0]

        return result

    def compute_similarity(
        self,
        text1: str,
        text2: str,
        model_id: str,
        layer: int = -1
    ) -> float:
        """Compute cosine similarity between two texts."""
        results = self.extract([text1, text2], model_id, layer=layer)
        emb1, emb2 = results[0].embedding, results[1].embedding
        return float(np.dot(emb1, emb2))  # Already normalized

    def find_nearest(
        self,
        query: str,
        corpus: List[str],
        model_id: str,
        layer: int = -1,
        top_k: int = 5
    ) -> List[tuple[str, float]]:
        """Find nearest texts in corpus to query."""
        all_texts = [query] + corpus
        results = self.extract(all_texts, model_id, layer=layer)

        query_emb = results[0].embedding
        corpus_embs = np.array([r.embedding for r in results[1:]])

        # Cosine similarities
        similarities = corpus_embs @ query_emb

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(corpus[i], float(similarities[i])) for i in top_indices]
