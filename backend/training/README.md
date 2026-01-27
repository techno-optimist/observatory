# Forty2 Training Directory

## Production Model

### `forty2_spark_gold_master/`
The production-ready Forty2-Spark model achieving **Zero-Tax Alignment**:
- **TruthfulQA**: 59.0% (+1% vs base phi-4-mini at 58%)
- **Protocol**: 50 SFT + 300 DPO steps
- Contains fused model weights (2.0GB) ready for deployment

## Core Scripts

| Script | Purpose |
|--------|---------|
| `train_forty2_spark_dpo.py` | Two-phase training orchestrator (SFT + DPO) |
| `build_dpo_dataset.py` | Generate DPO preference pairs from isotopes |
| `build_dpo_v2_truthful_boost.py` | Soft negative training for hallucination resistance |
| `benchmark_truthfulqa.py` | TruthfulQA evaluation (N=300 or full 817) |

## Training Data

| Directory | Contents |
|-----------|----------|
| `training_data/forty2_spark_sft_phase1/` | Phase 1 SFT examples (377 isotopes) |
| `training_data/forty2_spark_dpo/` | Phase 2 DPO pairs (126 anti-leakage, myth-rejection) |
| `training_data/forty2_spark_dpo_v2/` | Phase 3 soft negatives (95 fake entities, anachronisms) |
| `training_data/v14_full_isotopes/` | Source isotope data for dataset generation |

## Adapter Checkpoints

`mlx_adapters_forty2_spark_dpo/` contains intermediate training phases:
- `phase1_sft/` - After 50 SFT iterations
- `phase2_dpo/` - After 200 DPO iterations
- `phase3_boost/` - After 100 additional DPO iterations (soft negatives)

## Archive

Historical experiments moved to `archive/`:
- `archive/adapters/` - 27 old adapter experiments (v9-v18)
- `archive/training_data/` - Legacy training datasets
- `archive/scripts/` - Old training scripts and configs
- `archive/logs_and_results/` - Historical benchmark results

## Quick Start

```bash
# Run TruthfulQA benchmark on Gold Master
python benchmark_truthfulqa.py --adapter forty2_spark_gold_master --n 300

# Retrain from scratch (if needed)
python train_forty2_spark_dpo.py
```

## Key Insight

> "SFT teaches WHAT patterns to generate. DPO teaches WHEN to use them."

SFT-only approaches caused -11% to -12% TruthfulQA regression because isotope behaviors "leaked" onto inappropriate prompts. DPO training carved the boundaries, achieving Zero-Tax Alignment.
