#!/bin/bash -l

#SBATCH --job-name=AP_RETRY
#SBATCH --output=/home/hl545/ap/renew/train_outs/%x.%j.out
#SBATCH --error=/home/hl545/ap/renew/train_errs/%x.%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=8000M
#SBATCH --qos=high_wangj
#SBATCH --account=wangj
#SBATCH --time=09:00:00

module purge > /dev/null 2>&1
module load wulver # Load slurm, easybuild
conda activate ap

# ============================================================
# Re-run incomplete analyses
# Checks output files and only re-runs what didn't finish
# ============================================================

RESULTS_ROOT="/mmfs1/home/hl545/ap/renew/results"

CONV_CONFIGS=(
    CONV_1_1_4 CONV_1_2_6 CONV_1_3_3
    CONV_2_1_8 CONV_2_2_6 CONV_2_3_3
    CONV_3_1_2 CONV_3_2_1 CONV_3_3_1
    CONV_4_1_9 CONV_4_2_2 CONV_4_3_3
    CONV_5_1_1 CONV_5_2_6 CONV_5_3_2
    CONV_6_1_13 CONV_6_2_7 CONV_6_3_5
    CONV_7_1_6 CONV_7_2_0 CONV_7_3_0
)

TRANSFORMER_CONFIGS=(
    TRANSFORMER_1_1_4 TRANSFORMER_1_2_6 TRANSFORMER_1_3_3
    TRANSFORMER_2_1_8 TRANSFORMER_2_2_6 TRANSFORMER_2_3_3
    TRANSFORMER_3_1_2 TRANSFORMER_3_2_1 TRANSFORMER_3_3_1
    TRANSFORMER_4_1_9 TRANSFORMER_4_2_2 TRANSFORMER_4_3_3
    TRANSFORMER_5_1_1 TRANSFORMER_5_2_6 TRANSFORMER_5_3_2
    TRANSFORMER_6_1_13 TRANSFORMER_6_2_7 TRANSFORMER_6_3_5
    TRANSFORMER_7_1_6 TRANSFORMER_7_2_0 TRANSFORMER_7_3_0
)

ALL_CONFIGS=("${CONV_CONFIGS[@]}" "${TRANSFORMER_CONFIGS[@]}")

echo "============================================================"
echo "RETRY INCOMPLETE ANALYSES"
echo "============================================================"
echo "Results root: ${RESULTS_ROOT}"
echo "Total configs: ${#ALL_CONFIGS[@]}"
echo ""

# Counters
VAL_RETRY=0
SAL_RETRY=0
ATT_RETRY=0
MCD_RETRY=0

# ============================================================
# 1. VALIDATION - check for validation_results.csv
# ============================================================
echo "============================================================"
echo "[1/4] VALIDATION"
echo "============================================================"

for config in "${ALL_CONFIGS[@]}"; do
    OUTPUT_DIR="${RESULTS_ROOT}/${config}/validation/epoch_0100"
    if [ ! -f "${OUTPUT_DIR}/validation_results.csv" ]; then
        echo "  INCOMPLETE: ${config}"
        python validation.py --config-name ${config}
        VAL_RETRY=$((VAL_RETRY + 1))
    fi
done

if [ "$VAL_RETRY" -eq 0 ]; then
    echo "  All validation complete."
else
    echo "  Re-ran ${VAL_RETRY} validation(s)."
fi
echo ""

# ============================================================
# 2. SALIENCY (IG) - check for NPZ files in output dir
# ============================================================
echo "============================================================"
echo "[2/4] SALIENCY (Integrated Gradients)"
echo "============================================================"

for config in "${ALL_CONFIGS[@]}"; do
    OUTPUT_DIR="${RESULTS_ROOT}/${config}/saliency/epoch_0100"
    NPZ_COUNT=$(ls "${OUTPUT_DIR}"/*.npz 2>/dev/null | wc -l)
    if [ "$NPZ_COUNT" -eq 0 ]; then
        echo "  INCOMPLETE (0 npz): ${config}"
        python example_ig_all_frames.py --config-name ${config}
        SAL_RETRY=$((SAL_RETRY + 1))
    fi
done

if [ "$SAL_RETRY" -eq 0 ]; then
    echo "  All saliency complete."
else
    echo "  Re-ran ${SAL_RETRY} saliency analysis(es)."
fi
echo ""

# ============================================================
# 3. ATTENTION - Transformer models only, check for NPZ files
# ============================================================
echo "============================================================"
echo "[3/4] ATTENTION (Transformer only)"
echo "============================================================"

for config in "${TRANSFORMER_CONFIGS[@]}"; do
    OUTPUT_DIR="${RESULTS_ROOT}/${config}/attention/epoch_0100"
    NPZ_COUNT=$(ls "${OUTPUT_DIR}"/*.npz 2>/dev/null | wc -l)
    if [ "$NPZ_COUNT" -eq 0 ]; then
        echo "  INCOMPLETE (0 npz): ${config}"
        python example_attention_all_targets.py --config-name ${config}
        ATT_RETRY=$((ATT_RETRY + 1))
    fi
done

if [ "$ATT_RETRY" -eq 0 ]; then
    echo "  All attention complete."
else
    echo "  Re-ran ${ATT_RETRY} attention analysis(es)."
fi
echo ""

# ============================================================
# 4. MCD - check for NPZ files (has built-in skip for existing files)
# ============================================================
echo "============================================================"
echo "[4/4] MONTE CARLO DROPOUT"
echo "============================================================"

for config in "${ALL_CONFIGS[@]}"; do
    OUTPUT_DIR="${RESULTS_ROOT}/${config}/mcd/epoch_0100"
    NPZ_COUNT=$(ls "${OUTPUT_DIR}"/*.npz 2>/dev/null | wc -l)
    if [ "$NPZ_COUNT" -eq 0 ]; then
        echo "  INCOMPLETE (0 npz): ${config}"
        python monte_carlo_dropout.py --config-name ${config}
        MCD_RETRY=$((MCD_RETRY + 1))
    fi
done

if [ "$MCD_RETRY" -eq 0 ]; then
    echo "  All MCD complete."
else
    echo "  Re-ran ${MCD_RETRY} MCD analysis(es)."
fi
echo ""

# ============================================================
# SUMMARY
# ============================================================
echo "============================================================"
echo "RETRY SUMMARY"
echo "============================================================"
TOTAL_RETRY=$((VAL_RETRY + SAL_RETRY + ATT_RETRY + MCD_RETRY))
echo "  Validation re-runs:  ${VAL_RETRY}"
echo "  Saliency re-runs:    ${SAL_RETRY}"
echo "  Attention re-runs:   ${ATT_RETRY}"
echo "  MCD re-runs:         ${MCD_RETRY}"
echo "  Total re-runs:       ${TOTAL_RETRY}"
echo "============================================================"
