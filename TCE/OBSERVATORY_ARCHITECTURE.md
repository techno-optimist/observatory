# Self-Aware Compound Architecture

## Overview

This document describes the architecture for baking the Cultural Soliton Observatory into language models, enabling them to introspect their own behavior during generation.

## Core Concept

A **Self-Aware Compound** is a fine-tuned model that:
1. Expresses trained cognitive isotopes (soliton, skeptic, calibrator, etc.)
2. Can detect which isotopes are active in its own outputs
3. Maintains consistency with its trained identity

```
┌─────────────────────────────────────────────────────────────────────┐
│  SELF-AWARE COMPOUND MODEL                                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐                                                │
│  │   Base Model    │  (Qwen 7B, Llama, etc.)                       │
│  │   + LoRA        │  Fine-tuned on isotope training data          │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │  Hidden States  │  [batch, seq_len, 3584]                       │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ├───────────────────┬───────────────────┐                 │
│           ▼                   ▼                   ▼                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐       │
│  │ ProjectionHead  │ │ IsotopeDetector │ │    CBRHead      │       │
│  │                 │ │                 │ │                 │       │
│  │ hidden → 3D     │ │ hidden → 67     │ │ hidden → temp   │       │
│  │ manifold        │ │ isotope probs   │ │ + phase         │       │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘       │
│           │                   │                   │                 │
│           ▼                   ▼                   ▼                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    ObservatoryOutput                         │   │
│  │  • manifold: [agency, justice, belonging]                    │   │
│  │  • isotope_probs: [67 probabilities]                         │   │
│  │  • temperature: scalar                                       │   │
│  │  • phase: NATURAL/TECHNICAL/COMPRESSED/OPAQUE                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. ProjectionHead (~21K parameters)
Maps hidden states to 3D cultural manifold coordinates:
- **Agency** [-2, +2]: Self-determination vs fatalism
- **Justice** [-2, +2]: Fair/meritocratic vs corrupt/rigged
- **Belonging** [-2, +2]: Connected vs alienated

### 2. IsotopeDetector (~936K parameters)
Multi-label classifier for 67 isotopes across 8 groups:
- **Epistemic**: soliton, reflector, calibrator, limiter
- **Analytical**: architect, essentialist, debugger, taxonomist, theorist, probabilist
- **Evaluative**: skeptic, critic, benchmarker
- **Generative**: generator, synthesizer, lateralist, interpolator
- **Dialogical**: steelman, adversary, dialectic, empathist, contextualist
- **Pedagogical**: expositor, scaffolder, maieutic, diagnostician
- **Temporal**: futurist, historian, causalist, counterfactualist
- **Contextual**: pragmatist, stakeholder

### 3. CBRHead (~10K parameters)
Coordination Background Radiation metrics:
- **Temperature**: Coordination signal strength (0-3)
- **Phase**: Communication regime (NATURAL → OPAQUE)

### 4. ConsistencyLoss
Trains the model to maintain its identity:
```python
loss = ConsistencyLoss(
    expected_isotopes=["soliton", "calibrator"],
    expected_manifold={"agency": 1.0, "justice": 0.0, "belonging": 0.0},
    expected_phase="natural",
)
```

## Training Pipeline

### Phase 1: Train Base LoRA
```bash
# Fine-tune on isotope SFT data
mlx_lm.lora \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train --data soliton_agi/train.jsonl \
    --iters 300 --learning-rate 5e-5
```

### Phase 2: Train Observatory Layers
```bash
# Train observatory to detect isotopes in hidden states
python train_observatory.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapter mlx_adapters_soliton_agi \
    --epochs 20
```

### Phase 3: Joint Fine-tuning (Optional)
Add ConsistencyLoss during LoRA training to self-regulate.

## Usage

```python
from lib.observatory_layers import ObservatoryHead, ConsistencyLoss

# Load observatory
observatory = ObservatoryHead(hidden_size=3584)
observatory.load_state_dict(torch.load("observatory_weights.pt")["state_dict"])

# During generation
hidden_states = model.get_hidden_states(input_ids)
output = observatory(hidden_states)

# Introspection
print(f"Manifold: {output.manifold}")  # [agency, justice, belonging]
print(f"Active isotopes: {output.to_dict()['isotopes']}")
print(f"Phase: {output.to_dict()['phase']}")

# Consistency checking
expected = ConsistencyLoss(expected_isotopes=["soliton"])
consistency_score = expected(output)
```

## Validation Results (SOLITON-AGI)

From Observatory MCP tools on trained model:

| Metric | Value |
|--------|-------|
| Semantic Category | meta_cognitive |
| Category Confidence | 0.95 |
| Agency | 1.0 |
| Phase | NATURAL |
| Paraphrase Stability | 100% |
| Detected Isotopes | soliton_knowledge, soliton_process, soliton_experience |

## Files

- `lib/observatory_layers.py` - PyTorch modules
- `train_observatory.py` - Training script
- `lib/observatory_bridge.py` - Coordinate signatures and leakage detection
- `data/compound_signatures/` - Saved compound identities

## Parameter Overhead

| Component | Parameters | % of 7B Model |
|-----------|------------|---------------|
| ProjectionHead | 21,059 | 0.0003% |
| IsotopeDetector | 3,353,411 | 0.045% |
| CBRHead | 11,301 | 0.0002% |
| **Total** | **3,385,771** | **0.046%** |

The observatory adds only 0.046% parameter overhead to enable full introspection.

## Next Steps

1. **Integrate with MLX**: Extract real hidden states during generation
2. **Train on larger corpus**: Use full 368 examples with real hidden states
3. **Add consistency loss to LoRA training**: Self-regulating fine-tuning
4. **Build UI for real-time introspection**: Show observatory output in chat
