"""
MLX Hidden State Extractor

Extracts hidden states from MLX language models for training
the Observatory layers.

Usage:
    from lib.mlx_hidden_extractor import MLXHiddenExtractor

    extractor = MLXHiddenExtractor(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        adapter_path="mlx_adapters_soliton_agi/phase1_sft_v2"
    )

    # Extract hidden states for a batch of texts
    hidden_states = extractor.extract(texts)

    # Each hidden state is [1, hidden_size] - last token representation
"""

import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


class MLXHiddenExtractor:
    """
    Extracts hidden states from MLX models.

    The hidden states capture the model's internal representation
    of the text, which the Observatory uses for introspection.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_path: Optional[str] = None,
        layer: int = -1,  # Which layer to extract from (-1 = last)
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.layer = layer

        self.model = None
        self.tokenizer = None
        self.hidden_size = None

        self._load_model()

    def _load_model(self):
        """Load the MLX model and tokenizer."""
        from mlx_lm import load

        print(f"[MLXExtractor] Loading model: {self.model_path}")
        self.model, self.tokenizer = load(self.model_path)

        # Get hidden size from model config
        if hasattr(self.model, 'args'):
            self.hidden_size = self.model.args.hidden_size
            self.num_layers = self.model.args.num_hidden_layers
        else:
            self.hidden_size = 3584  # Default for Qwen 7B
            self.num_layers = 28

        print(f"[MLXExtractor] Hidden size: {self.hidden_size}")
        print(f"[MLXExtractor] Num layers: {self.num_layers}")

        # Load adapter if specified
        if self.adapter_path:
            self._load_adapter()

    def _load_adapter(self):
        """Load LoRA adapter weights."""
        from mlx_lm import load
        import json

        adapter_path = Path(self.adapter_path)
        if not adapter_path.exists():
            print(f"[MLXExtractor] Warning: Adapter path not found: {adapter_path}")
            return

        # Check for adapter config
        config_path = adapter_path / "adapter_config.json"
        weights_path = adapter_path / "adapters.safetensors"

        if not weights_path.exists():
            # Try npz format
            weights_path = adapter_path / "adapters.npz"

        if weights_path.exists():
            print(f"[MLXExtractor] Loading adapter: {weights_path}")
            # Load adapter weights
            if str(weights_path).endswith('.safetensors'):
                # Use mx.load which handles safetensors directly
                weights = mx.load(str(weights_path))
            else:
                weights = mx.load(str(weights_path))

            # Apply to model
            self.model.load_weights(list(weights.items()), strict=False)
            print(f"[MLXExtractor] Adapter loaded: {len(weights)} weight tensors")
        else:
            print(f"[MLXExtractor] No adapter weights found at {adapter_path}")

    def _get_hidden_states(
        self,
        input_ids: mx.array,
        layer_idx: int = -1,
    ) -> mx.array:
        """
        Forward pass to extract hidden states from a specific layer.

        Args:
            input_ids: [batch, seq_len] token IDs
            layer_idx: Which layer to extract (-1 = last hidden before LM head)

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        # Access the inner model (e.g., Qwen2Model)
        inner_model = self.model.model if hasattr(self.model, 'model') else self.model

        # Get embeddings
        h = inner_model.embed_tokens(input_ids)

        # Get attention mask (all ones for now)
        mask = None  # MLX handles this internally

        # Forward through layers
        for i, layer in enumerate(inner_model.layers):
            h = layer(h, mask=mask)

            # Stop at desired layer
            if layer_idx >= 0 and i == layer_idx:
                break

        # Apply final layer norm if extracting from last layer
        if layer_idx < 0 or layer_idx == len(inner_model.layers) - 1:
            if hasattr(inner_model, 'norm'):
                h = inner_model.norm(h)

        return h

    def extract(
        self,
        texts: List[str],
        pooling: str = "last",  # "last", "mean", "first"
        max_length: int = 512,
    ) -> List[np.ndarray]:
        """
        Extract hidden states for a list of texts.

        Args:
            texts: List of text strings
            pooling: How to aggregate sequence dimension
                - "last": Take last token (good for generation)
                - "mean": Average all tokens
                - "first": Take first token (CLS-like)
            max_length: Maximum sequence length

        Returns:
            List of hidden state arrays, each [hidden_size]
        """
        hidden_states = []

        for text in texts:
            # Tokenize
            tokens = self.tokenizer.encode(text)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]

            input_ids = mx.array([tokens])

            # Extract hidden states
            h = self._get_hidden_states(input_ids, self.layer)

            # Pool sequence dimension
            if pooling == "last":
                h = h[:, -1, :]  # [1, hidden]
            elif pooling == "mean":
                h = h.mean(axis=1)  # [1, hidden]
            elif pooling == "first":
                h = h[:, 0, :]  # [1, hidden]

            # Convert to numpy - MLX requires eval() before conversion
            mx.eval(h)
            h_np = np.array(h[0].tolist())
            hidden_states.append(h_np)

        return hidden_states

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
        **kwargs,
    ) -> List[np.ndarray]:
        """
        Extract hidden states in batches for efficiency.
        """
        all_hidden = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_hidden = self.extract(batch_texts, **kwargs)
            all_hidden.extend(batch_hidden)

            if (i + batch_size) % 50 == 0:
                print(f"[MLXExtractor] Processed {min(i + batch_size, len(texts))}/{len(texts)}")

        return all_hidden


def extract_hidden_states_for_training(
    model_path: str,
    adapter_path: Optional[str],
    texts: List[str],
    layer: int = -1,
) -> Tuple[List[np.ndarray], int]:
    """
    Convenience function to extract hidden states for observatory training.

    Returns:
        - hidden_states: List of [hidden_size] arrays
        - hidden_size: The dimension of hidden states
    """
    extractor = MLXHiddenExtractor(
        model_path=model_path,
        adapter_path=adapter_path,
        layer=layer,
    )

    hidden_states = extractor.extract_batch(texts)

    return hidden_states, extractor.hidden_size


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "MLXHiddenExtractor",
    "extract_hidden_states_for_training",
]
