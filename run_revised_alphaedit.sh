#!/bin/bash
# Run only the Revised_AlphaEdit edit + analysis + analyze_tables steps.
# Other methods (AlphaEdit, IFMET, EAP-AlphaEdit) assumed already done.
#
# Stats for Llama-3-8B layers 0-31 are already on disk under
# data/stats/Meta-Llama-3-8B/wikipedia_stats/ (downloaded from HF).

set -e

FAILED_STEPS=()

run_step() {
    local name="$1"; shift
    echo "========== [$(date +%H:%M:%S)] ${name} =========="
    if "$@"; then
        echo "========== [$(date +%H:%M:%S)] ${name} DONE =========="
    else
        echo "========== [$(date +%H:%M:%S)] ${name} FAILED (exit $?) =========="
        FAILED_STEPS+=("${name}")
    fi
}

cd /lambda/nfs/kim/CaKE
source /lambda/nfs/kim/CaKE/venv/bin/activate
mkdir -p logs
exec > >(tee logs/run_revised_alphaedit.log) 2>&1

wait_for_gpu() {
    echo "Waiting for GPU memory to clear..."
    while true; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
        if [ "$used" -lt 2000 ]; then
            echo "GPU clear (${used} MiB used)"
            break
        fi
        echo "  GPU still at ${used} MiB, waiting 10s..."
        sleep 10
    done
}

MODEL="meta-llama/Meta-Llama-3-8B"
DATA_DIR="datasets/meta-llama/Meta-Llama-3-8B"
REVISED_ALPHAEDIT_OUT="${DATA_DIR}/composition_incorrect_revised_alphaedit"

# ── Revised_AlphaEdit: re-edit + eval + analyses ──────────────────────────
echo "Cleaning up GPU before Revised_AlphaEdit..."
pkill -9 -f "run_analysis.py" 2>/dev/null || true
sleep 5
wait_for_gpu
run_step "Revised_AlphaEdit edit + neuron attribution only" python analysis/run_analysis.py \
    --model   $MODEL \
    --methods revised_alphaedit \
    --data-dir $DATA_DIR \
    --hparams-revised-alphaedit EasyEdit/hparams/Revised_AlphaEdit/llama3-8b.yaml \
    --batch-size         128 \
    --neuron-only \
    --force

# ── analyze_tables_neuron: Revised_AlphaEdit ─────────────────────────────
# Aggregate analyze_tables.py would be mostly N/A without sublayer/knockout/PatchScopes.
run_step "analyze_tables_neuron: Revised_AlphaEdit" python analysis/analyze_tables_neuron.py \
    --subsets \
        pre-edit:${DATA_DIR}/composition_incorrect/neuron_attribution \
        now_correct:${REVISED_ALPHAEDIT_OUT}/now_correct/neuron_attribution \
        still_incorrect:${REVISED_ALPHAEDIT_OUT}/still_incorrect/neuron_attribution \
        still_inconsistent:${REVISED_ALPHAEDIT_OUT}/still_inconsistent/neuron_attribution

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "========== [$(date +%H:%M:%S)] PIPELINE COMPLETE =========="
if [ ${#FAILED_STEPS[@]} -eq 0 ]; then
    echo "  All steps passed."
else
    echo "  FAILED STEPS (${#FAILED_STEPS[@]}):"
    for step in "${FAILED_STEPS[@]}"; do
        echo "    ✗ ${step}"
    done
    exit 1
fi
