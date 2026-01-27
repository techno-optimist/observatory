# Projection Modes Guide

**Cultural Soliton Observatory**
**Version:** January 2026

---

## Overview

The Cultural Soliton Observatory supports four projection configurations, each optimized for different use cases. This guide explains when to use each mode, their trade-offs, and how to select them via the API.

| Mode | Embedding Model | Key Strength | Best For |
|------|-----------------|--------------|----------|
| `current_projection` | all-MiniLM-L6-v2 | Fast, lightweight | Quick exploration |
| `mpnet_projection` | all-mpnet-base-v2 | Best single-model accuracy | Production analysis |
| `multi_model_ensemble` | MiniLM + MPNet + paraphrase-MiniLM | Best robustness | High-stakes decisions |
| `ensemble_projection` | 25 bootstrap models | Uncertainty quantification | Research applications |

---

## 1. Current Projection (Default Baseline)

### Model Configuration
- **Embedding Model:** `all-MiniLM-L6-v2` (384 dimensions)
- **Projection:** Ridge regression
- **CV Score:** 0.383
- **R-squared:** 0.576

### How It Works
The default projection uses MiniLM-L6-v2, a compact sentence transformer model (22M parameters). Embeddings are projected to the 3D cultural manifold (agency, perceived_justice, belonging) using a trained Ridge regression model.

### When to Use
- **Quick exploration** and prototyping
- **Resource-constrained environments** (CPU-only, limited memory)
- **High-throughput batch processing** where speed matters more than accuracy
- **Backward compatibility** with existing analyses

### Limitations
- Lowest accuracy among projection modes
- Paraphrase robustness concerns (validity study showed 0.926 mean spread)
- May capture surface linguistic features rather than deep semantics

### API Usage

```python
import requests

# Default analyze endpoint uses current_projection
response = requests.post("http://localhost:8000/analyze", json={
    "text": "I believe we can build a fair society together.",
    "model_id": "all-MiniLM-L6-v2"
})

result = response.json()
# Returns: vector, mode, confidence, embedding_dim, layer, model_id
```

```bash
# cURL example
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Hard work leads to success.", "model_id": "all-MiniLM-L6-v2"}'
```

---

## 2. MPNet Projection (Best Single-Model Accuracy)

### Model Configuration
- **Embedding Model:** `all-mpnet-base-v2` (768 dimensions)
- **Projection:** Ridge regression (retrained for MPNet dimensions)
- **CV Score:** 0.612 (60% improvement over default)
- **R-squared:** 0.775
- **Test R-squared:** 0.661

### How It Works
Uses MPNet-base-v2, a more powerful sentence transformer (110M parameters) that produces richer 768-dimensional embeddings. The projection is separately trained on these embeddings to leverage the improved semantic representation.

### When to Use
- **Production analysis** where accuracy is critical
- **Single-text analysis** where latency is acceptable (~2x slower than MiniLM)
- **Research** requiring high-fidelity semantic measurement
- When you need the **best point estimate** without uncertainty bounds

### Limitations
- Slower inference than MiniLM (~2x)
- Higher memory usage (768-dim vs 384-dim)
- Still has robustness concerns from validity study
- No built-in uncertainty quantification

### API Usage

```python
import requests

# Use MPNet model for higher accuracy
response = requests.post("http://localhost:8000/analyze", json={
    "text": "The system rewards those who work hard and play by the rules.",
    "model_id": "all-mpnet-base-v2"
})

# Note: Requires projection trained with MPNet embeddings
# Train with:
# POST /training/train {"model_id": "all-mpnet-base-v2", "method": "ridge"}
```

### Training MPNet Projection

```python
# Train a projection specifically for MPNet embeddings
response = requests.post("http://localhost:8000/training/train", json={
    "model_id": "all-mpnet-base-v2",
    "method": "ridge",
    "auto_tune_alpha": True
})

# Response includes metrics showing the improvement
print(response.json()["metrics"])
# Expected: cv_score_mean ~0.612, r2_overall ~0.775
```

---

## 3. Multi-Model Ensemble (Best Robustness)

### Model Configuration
- **Embedding Models:**
  - `all-MiniLM-L6-v2` (384 dimensions)
  - `all-mpnet-base-v2` (768 dimensions)
  - `paraphrase-MiniLM-L12-v2` (384 dimensions)
- **Paraphrase Spread:** 0.724 (best robustness metric)
- **Aggregation:** Weighted average of projections

### How It Works
Projects text through three different embedding models, then averages the resulting coordinates. Each model captures different aspects of semantics:

1. **MiniLM-L6-v2:** Fast, general-purpose embeddings
2. **MPNet-base-v2:** High-quality semantic representations
3. **Paraphrase-MiniLM-L12-v2:** Optimized for paraphrase detection

The ensemble reduces sensitivity to specific model biases and linguistic surface features.

### When to Use
- **High-stakes decisions** requiring robust classifications
- **Texts with potential paraphrasing** or varied phrasings
- When the **same meaning expressed differently** should yield similar projections
- **Validation studies** comparing model agreement
- When robustness matters more than speed

### Limitations
- Slowest projection mode (~3x slower than single-model)
- Requires all three models loaded in memory
- No uncertainty quantification (reports disagreement instead)
- Disagreement between models may indicate text ambiguity or projection instability

### API Usage

```python
import requests

# Multi-model analysis endpoint
response = requests.post("http://localhost:8000/analyze/multi-model", json={
    "text": "Everyone has the power to change their circumstances.",
    "model_ids": [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-MiniLM-L12-v2"
    ]
})

result = response.json()

# Returns per-model projections and ensemble agreement metrics
print(result["per_model"])  # Individual model results
print(result["agreement"]["ensemble_projection"])  # Averaged projection
print(result["agreement"]["disagreement"])  # Std dev across models
print(result["agreement"]["embedding_similarities"])  # Pairwise cosine similarities
```

### Interpreting Disagreement

```python
# High disagreement indicates unstable projection
disagreement = result["agreement"]["disagreement"]

if max(disagreement.values()) > 0.5:
    print("Warning: High model disagreement - projection may be unreliable")
    print("Consider using ensemble_projection with uncertainty bounds instead")
```

---

## 4. Ensemble Projection (Uncertainty Quantification)

### Model Configuration
- **Architecture:** 25 Ridge regression models
  - 5 regularization strengths: alpha = [0.01, 0.1, 1.0, 10.0, 100.0]
  - 5 bootstrap samples per alpha
- **Mean R-squared:** 0.971 (individual model average)
- **Ensemble R-squared:** 0.833 (combined prediction)
- **Output:** Mean projection + confidence intervals

### How It Works
Trains 25 Ridge regression models with different regularization strengths and bootstrap samples of the training data. For inference:

1. Each model produces a projection
2. Mean and standard deviation computed across 25 predictions
3. 95% confidence intervals calculated per axis
4. Overall confidence score derived from uncertainty magnitude

This captures both **model uncertainty** (different regularizations) and **data uncertainty** (bootstrap sampling).

### When to Use
- **Research applications** requiring uncertainty estimates
- When you need to know **how confident** the projection is
- **Boundary case detection** (ambiguous classifications)
- **Scientific publications** requiring error bounds
- When making **decisions based on projection values**

### Output Fields

| Field | Description |
|-------|-------------|
| `coords` | Mean projection (agency, perceived_justice, belonging) |
| `std_per_axis` | Standard deviation per axis |
| `confidence_intervals` | 95% CI per axis: [lower, upper] |
| `overall_confidence` | Single confidence score (0-1) |
| `method` | "ensemble_ridge" |

### API Usage

#### Training the Ensemble

```python
import requests

# Train ensemble projection (one-time setup)
response = requests.post("http://localhost:8000/training/train-ensemble", json={
    "model_id": "all-MiniLM-L6-v2",
    "n_bootstrap": 5,
    "alphas": [0.01, 0.1, 1.0, 10.0, 100.0]  # Optional, these are defaults
})

result = response.json()
print(f"Trained {result['ensemble_size']} models")
print(f"Ensemble R-squared: {result['metrics']['ensemble_r2']:.3f}")
```

#### Using the Ensemble

```python
# v2 analyze endpoint automatically uses ensemble if trained
response = requests.post("http://localhost:8000/v2/analyze", json={
    "text": "Sometimes I feel powerful, other times helpless.",
    "model_id": "all-MiniLM-L6-v2",
    "include_uncertainty": True
})

result = response.json()

# Access uncertainty information
uncertainty = result["uncertainty"]
print(f"Agency: {result['vector']['agency']:.2f} +/- {uncertainty['std_per_axis']['agency']:.2f}")
print(f"95% CI: [{uncertainty['confidence_intervals']['agency'][0]:.2f}, "
      f"{uncertainty['confidence_intervals']['agency'][1]:.2f}]")
print(f"Overall confidence: {uncertainty['overall_confidence']:.3f}")
```

### Interpreting Confidence Intervals

```python
# Wide confidence intervals indicate uncertain projections
ci = result["uncertainty"]["confidence_intervals"]

for axis, (lower, upper) in ci.items():
    width = upper - lower
    if width > 1.0:
        print(f"Warning: Wide CI for {axis} ({width:.2f}) - high uncertainty")
    elif width > 0.5:
        print(f"Moderate uncertainty for {axis} ({width:.2f})")
    else:
        print(f"Confident projection for {axis} ({width:.2f})")
```

---

## Performance Comparison

### Accuracy Metrics

| Projection Mode | CV Score | R-squared | Test R-squared | Notes |
|-----------------|----------|-----------|----------------|-------|
| current_projection | 0.383 | 0.576 | - | Baseline |
| mpnet_projection | 0.612 | 0.775 | 0.661 | 60% better CV |
| multi_model_ensemble | - | - | - | Best robustness |
| ensemble_projection | - | 0.971* | 0.833 | *Mean individual |

### Robustness Metrics

| Projection Mode | Paraphrase Spread | Mode Consistency | Adversarial Robustness |
|-----------------|-------------------|------------------|------------------------|
| current_projection | 0.926 (high) | 0/4 concepts | 0.698 |
| mpnet_projection | ~0.8 (estimated) | - | - |
| multi_model_ensemble | **0.724** (best) | Improved | - |
| ensemble_projection | Varies | Reports uncertainty | - |

*Lower paraphrase spread = better (semantically equivalent texts project closer)*

### Speed and Resource Usage

| Projection Mode | Relative Speed | Memory (typical) | GPU Benefit |
|-----------------|---------------|------------------|-------------|
| current_projection | 1.0x (fastest) | ~500MB | Moderate |
| mpnet_projection | 0.5x | ~1GB | Significant |
| multi_model_ensemble | 0.3x | ~2GB | Significant |
| ensemble_projection | 0.8x | ~600MB | Low |

---

## Use Case Decision Tree

```
Need projection for text analysis?
|
+-- Speed is critical?
|   |
|   +-- YES --> current_projection (MiniLM)
|   +-- NO --> Continue...
|
+-- Need uncertainty bounds?
|   |
|   +-- YES --> ensemble_projection (25 bootstrap models)
|   +-- NO --> Continue...
|
+-- Concerned about paraphrase sensitivity?
|   |
|   +-- YES --> multi_model_ensemble (3 models)
|   +-- NO --> Continue...
|
+-- Best single-point accuracy needed?
    |
    +-- YES --> mpnet_projection
    +-- NO --> current_projection (default)
```

---

## Detailed Use Case Scenarios

### 1. Research Applications

**Recommended:** `ensemble_projection`

```python
# Research workflow with proper uncertainty reporting
response = requests.post("http://localhost:8000/v2/analyze", json={
    "text": research_text,
    "include_uncertainty": True
})

result = response.json()

# Report with confidence intervals
print(f"Agency: {result['vector']['agency']:.3f} "
      f"(95% CI: {result['uncertainty']['confidence_intervals']['agency']})")
```

**Why:** Research requires understanding measurement uncertainty. The validity study found that 87.5% of ambiguous statements have low confidence (<0.5), and 43.2% of perturbations can cause mode flips. Reporting uncertainty is essential for scientific validity.

### 2. Production/Real-time Analysis

**Recommended:** `mpnet_projection`

```python
# Production analysis with MPNet
response = requests.post("http://localhost:8000/analyze", json={
    "text": user_submitted_text,
    "model_id": "all-mpnet-base-v2"
})

# 60% better accuracy than default, acceptable latency
```

**Why:** MPNet offers the best accuracy for single-model inference. The 2x latency increase is acceptable for most production workloads, and the 60% improvement in CV score significantly improves reliability.

### 3. High-Stakes Decisions Requiring Robustness

**Recommended:** `multi_model_ensemble`

```python
# High-stakes analysis with robustness checking
response = requests.post("http://localhost:8000/analyze/multi-model", json={
    "text": important_text,
    "model_ids": [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-MiniLM-L12-v2"
    ]
})

result = response.json()

# Check for model agreement before acting on result
max_disagreement = max(result["agreement"]["disagreement"].values())
if max_disagreement > 0.3:
    raise Warning("Models disagree significantly - manual review recommended")
```

**Why:** The validity study found low paraphrase robustness (0.926 spread). Multi-model ensemble achieves 0.724 spread - the best robustness among all modes. For decisions with real consequences, model agreement provides an additional validity check.

### 4. When Uncertainty Estimates Are Needed

**Recommended:** `ensemble_projection`

```python
# Train ensemble (one-time)
requests.post("http://localhost:8000/training/train-ensemble", json={
    "model_id": "all-MiniLM-L6-v2"
})

# Analyze with uncertainty
response = requests.post("http://localhost:8000/v2/analyze", json={
    "text": "The deck is stacked against ordinary people.",
    "include_uncertainty": True
})

result = response.json()

# Make decisions based on confidence
if result["uncertainty"]["overall_confidence"] < 0.5:
    # High uncertainty - classification may be unreliable
    flag_for_review(result)
else:
    # Confident classification
    process_normally(result)
```

**Why:** Only ensemble_projection provides calibrated uncertainty estimates. The 25-model ensemble captures both model uncertainty (different regularizations) and data uncertainty (bootstrap sampling).

---

## Technical Details

### How Each Mode Works

#### Current Projection (Ridge Regression)
```
Text -> MiniLM Embedding (384d) -> Scaler -> Ridge Model -> 3D Coordinates
                                              |
                                              +-- Trained on 64+ examples
                                              +-- Single alpha value (auto-tuned)
```

#### MPNet Projection
```
Text -> MPNet Embedding (768d) -> Scaler -> Ridge Model -> 3D Coordinates
                                             |
                                             +-- Same architecture, different weights
                                             +-- Leverages richer embeddings
```

#### Multi-Model Ensemble
```
Text -> MiniLM Embedding -> Scaler -> Ridge -> Coords_1 --+
     |                                                     |
     +-> MPNet Embedding -> Scaler -> Ridge -> Coords_2 --+-> Weighted Average
     |                                                     |
     +-> Paraphrase-MiniLM -> Scaler -> Ridge -> Coords_3 -+

     Also computes: model disagreement (std dev), embedding similarities
```

#### Ensemble Projection
```
                              +-> Ridge (alpha=0.01, bootstrap_1) --+
                              +-> Ridge (alpha=0.01, bootstrap_2) --+
                              +-> ...                               |
Text -> MiniLM Embedding ->   +-> Ridge (alpha=0.1, bootstrap_1) ---+-> Mean, Std, CI
                              +-> ...                               |
                              +-> Ridge (alpha=100.0, bootstrap_5) -+

                              (25 models total: 5 alphas x 5 bootstraps)
```

### Embedding Model Specifications

| Model | Dimensions | Parameters | Max Tokens | Pooling |
|-------|------------|------------|------------|---------|
| all-MiniLM-L6-v2 | 384 | 22M | 256 | Mean |
| all-mpnet-base-v2 | 768 | 110M | 384 | Mean |
| paraphrase-MiniLM-L12-v2 | 384 | 33M | 128 | Mean |

### Ridge Regression Configuration

- **Auto-tuned alphas:** [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
- **Selection method:** 5-fold cross-validation
- **Normalization:** StandardScaler on inputs
- **Output:** 3 values (agency, perceived_justice, belonging)

---

## API Reference Summary

### Projection Selection

| Endpoint | Default Mode | Override |
|----------|--------------|----------|
| `POST /analyze` | current_projection | Change `model_id` |
| `POST /v2/analyze` | current_projection + ensemble uncertainty | `model_id`, `include_uncertainty` |
| `POST /analyze/multi-model` | multi_model_ensemble | Specify `model_ids` |
| `POST /projection/project-with-method` | Specified method | `method` param |

### Check Ensemble Status

```python
# Check if ensemble is trained and available
response = requests.get("http://localhost:8000/training/ensemble-status")
print(response.json())
# {"trained": true, "ensemble_size": 25, "metrics": {...}}
```

### Compare All Methods

```python
# Compare performance of all projection methods on current training data
response = requests.post("http://localhost:8000/projection/compare", json={
    "model_id": "all-MiniLM-L6-v2",
    "methods": ["ridge", "gp", "cav"]
})

# Returns comparative metrics and best method recommendation
print(response.json()["summary"]["best_method"])
```

---

## Recommendations

### General Guidelines

1. **Start with `current_projection`** for exploration and development
2. **Use `mpnet_projection`** for production when accuracy matters
3. **Use `multi_model_ensemble`** when robustness is critical
4. **Always use `ensemble_projection`** for research and publications

### When in Doubt

- **Check confidence:** If `overall_confidence < 0.5`, the classification is uncertain
- **Check boundary cases:** If `is_boundary_case: true`, small changes could flip the mode
- **Check model agreement:** High disagreement in multi-model indicates text ambiguity

### Validity Considerations

From the January 2026 Validity Study:
- Paraphrase robustness is **LOW** (0.926 spread) - same meaning can project differently
- Mode boundaries are **SENSITIVE** (87.5% ambiguous statements show low confidence)
- Axes are **PARTIALLY ENTANGLED** (66.3% specificity to target axis)
- Always report uncertainty when using projections for decisions

---

## Changelog

- **January 2026:** Added ensemble_projection with uncertainty quantification
- **January 2026:** Added multi_model_ensemble for improved robustness
- **January 2026:** Renamed "fairness" axis to "perceived_justice"
- **January 2026:** Completed validity study informing these recommendations
