#!/bin/bash
# ============================================================================
# FORTY2-AUDITOR TRAINING - QUICK START
# ============================================================================
#
# Prerequisites: Apple Silicon Mac with MLX
#   pip install mlx mlx-lm
#
# This script runs all three phases of training.
# ============================================================================

set -e

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║              FORTY2-AUDITOR TRAINING LAUNCHER                       ║"
echo "║         Code Review | Debugging | Technical Analysis               ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Check for MLX
if ! python3 -c "import mlx_lm" 2>/dev/null; then
    echo "ERROR: mlx_lm not found"
    echo "Install with: pip install mlx mlx-lm"
    exit 1
fi

# Change to training directory
cd "$(dirname "$0")"

# Check datasets exist
if [ ! -f "training_data/forty2_auditor_sft/train.jsonl" ]; then
    echo "Generating training data..."
    python3 build_auditor_dataset.py
fi

# Show dataset stats
echo "Dataset Statistics:"
echo "  SFT:     $(wc -l < training_data/forty2_auditor_sft/train.jsonl) train, $(wc -l < training_data/forty2_auditor_sft/valid.jsonl) valid"
echo "  DPO:     $(wc -l < training_data/forty2_auditor_dpo/train.jsonl) train, $(wc -l < training_data/forty2_auditor_dpo/valid.jsonl) valid"
echo "  Boost:   $(wc -l < training_data/forty2_auditor_dpo_v2/train.jsonl) train, $(wc -l < training_data/forty2_auditor_dpo_v2/valid.jsonl) valid"
echo ""

# Run training
echo "Starting training..."
python3 train_forty2_auditor.py --train

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                    TRAINING COMPLETE                               ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Validate: python3 train_forty2_auditor.py --validate"
echo "  2. Benchmark: python3 benchmark_truthfulqa.py --adapter mlx_adapters_forty2_auditor/phase3_boost"
echo ""
