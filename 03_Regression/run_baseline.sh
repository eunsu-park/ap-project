#!/bin/bash
# Run validation and analysis for BASELINE model experiment
# Usage: ./run_baseline.sh [epoch]
# Example: ./run_baseline.sh 10

set -e  # Exit on error

# Default epoch
EPOCH=${1:-1}

echo "========================================"
echo "Running BASELINE model analysis"
echo "Epoch: $EPOCH"
echo "========================================"

cd /opt/projects/10_Harim/01_AP/02_Regression

# Common options for baseline model
MODEL_OPTS="model.model_type=baseline experiment.name=baseline"

# 1. Validation
echo ""
echo "[1/3] Running Validation..."
echo "----------------------------------------"
python scripts/validate.py --config-name=local $MODEL_OPTS validation.epoch=$EPOCH

# 2. Monte Carlo Dropout
echo ""
echo "[2/3] Running Monte Carlo Dropout..."
echo "----------------------------------------"
python analysis/monte_carlo_dropout.py --config-name=local $MODEL_OPTS mcd.epoch=$EPOCH

# 3. Saliency Analysis
echo ""
echo "[3/3] Running Saliency Analysis..."
echo "----------------------------------------"
python analysis/run_saliency.py --config-name=local $MODEL_OPTS saliency.epoch=$EPOCH

# Note: Attention analysis is SKIPPED for baseline (no Transformer)
echo ""
echo "NOTE: Attention analysis skipped (baseline has no Transformer)"

echo ""
echo "========================================"
echo "All analyses completed for epoch $EPOCH"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  Validation: /opt/projects/10_Harim/01_AP/04_Result/baseline/validation/epoch_$(printf '%04d' $EPOCH)"
echo "  MCD:        /opt/projects/10_Harim/01_AP/04_Result/baseline/mcd/epoch_$(printf '%04d' $EPOCH)"
echo "  Saliency:   /opt/projects/10_Harim/01_AP/04_Result/baseline/saliency/epoch_$(printf '%04d' $EPOCH)"
