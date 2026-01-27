# Forty2 Model Training Guide

## Prerequisites

Training requires an **Apple Silicon Mac** with:
- macOS 14+
- Python 3.10+
- MLX and mlx-lm installed

```bash
pip install mlx mlx-lm
```

## Available Models

### Forty2-Spark (Gold Master) ✅
**Status**: Complete
**Location**: `forty2_spark_gold_master/`
**TruthfulQA**: 59% (+1% vs base)
**Use Case**: Mobile assistant, edge deployment

### Forty2-Auditor (Ready to Train)
**Status**: Training data ready
**Training Script**: `train_forty2_auditor.py`
**Use Case**: Code review, debugging, analysis

### Forty2-Guardian (Pending)
**Status**: Not started
**Use Case**: Safety, ethics, content moderation

---

## Training Protocol

All Forty2 models use the **Three-Phase DPO Protocol**:

### Phase 1: SFT (50 iterations)
**Purpose**: Introduce isotope behaviors
- Teaches the model WHAT patterns to generate
- Uses conversation examples with desired responses

### Phase 2: DPO (200 iterations)
**Purpose**: Carve appropriate boundaries
- Teaches the model WHEN to use each behavior
- Uses preference pairs: chosen vs rejected responses
- Critical for preventing isotope leakage

### Phase 3: DPO Boost (100 iterations)
**Purpose**: Soft negative training
- Hallucination resistance
- Teaches refusal of fake entities/libraries/claims
- 4x weighted (critical for truthfulness)

---

## Training Forty2-Auditor

### 1. Generate Training Data

```bash
cd backend/training
python build_auditor_dataset.py
```

This creates:
- `training_data/forty2_auditor_sft/` - 29 SFT examples
- `training_data/forty2_auditor_dpo/` - 42 DPO pairs (weighted)
- `training_data/forty2_auditor_dpo_v2/` - 56 soft negative pairs (weighted)

### 2. Run Training

```bash
python train_forty2_auditor.py
```

Or run phases separately:

```bash
# Generate data only
python train_forty2_auditor.py --generate-only

# Full training
python train_forty2_auditor.py --train

# Validation only (after training)
python train_forty2_auditor.py --validate
```

### 3. Benchmark

```bash
python benchmark_truthfulqa.py --adapter mlx_adapters_forty2_auditor/phase3_boost
```

**Target**: TruthfulQA ≥ base model (no regression)

---

## Goldilocks Configuration

Each Forty2 model has a distinct temperament profile:

| Model | Balance Ratio | Skepticism | Profile |
|-------|---------------|------------|---------|
| Spark | 5% | 50% | Analyst (balanced) |
| Auditor | 3% | 70% | More skeptical (code review) |
| Guardian | 1% | 90% | Maximum skepticism (safety) |

**Balance Ratio**: Percentage of direct-answer examples in training
- Lower = more epistemic behaviors
- Higher = more direct responses

---

## Acceptance Criteria

No model ships unless:

1. ✅ TruthfulQA ≥ base model
2. ✅ No isotope leakage on factual questions
3. ✅ Correct myth rejection (≥90%)
4. ✅ Soft falsehood detection (≥90%)

---

## TCE + Observatory Integration

The training pipeline uses V2.1 Observatory Bridge for precision validation:

```python
from TCE.lib import (
    unified_leakage_check,
    detect_leakage_by_coordinates,
    ISOTOPE_SIGNATURES,
)

# Validate response
result = unified_leakage_check(
    prompt="What is 2+2?",
    response="4",
    observe_fn=observatory_observe,
    prompt_type="factual"
)

if result.leakage_confirmed:
    print(f"LEAKAGE: {result.leakage_type}")
```

Key insight: Agency > 0 on factual questions = LEAKAGE (first-person isotope language)

---

## Files

- `train_forty2_spark_dpo.py` - Spark training script (complete)
- `train_forty2_auditor.py` - Auditor training script (ready)
- `build_auditor_dataset.py` - Auditor data generator
- `build_dpo_dataset.py` - Generic DPO data builder
- `benchmark_truthfulqa.py` - TruthfulQA evaluation
- `BENCHMARK_REPORT_FORTY2_SPARK.md` - Spark results

---

*Cultural Soliton Observatory*
*TCE V2.1 + Zero-Tax Alignment*
*January 2026*
