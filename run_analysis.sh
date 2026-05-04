#!/bin/bash
# Full pipeline: re-edit with AlphaEdit, IFMET, and EAP-AlphaEdit (fresh checkpoints),
# run analyses, analyze_tables.
#
# Batch sizes (tuned for A100 40GB + Llama-3-8B):
#   --batch-size 128  (unified: inference, edit outer loop, and edit→analyze→restore cycles)

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
exec > >(tee logs/run_analysis.log) 2>&1

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

# ── 0. Precompute covariance + null-space projection P (shared by AlphaEdit & EAP-AlphaEdit)
run_step "Precompute covariance + P" python precompute_P.py

MODEL="meta-llama/Meta-Llama-3-8B"
DATA_DIR="datasets/meta-llama/Meta-Llama-3-8B"
IFMET_OUT="${DATA_DIR}/composition_incorrect_ifmet"
ALPHAEDIT_OUT="${DATA_DIR}/composition_incorrect_alphaedit"
EAP_ALPHAEDIT_OUT="${DATA_DIR}/composition_incorrect_eap_alphaedit"
REVISED_ALPHAEDIT_OUT="${DATA_DIR}/composition_incorrect_revised_alphaedit"
ALL_LAYERS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"

# ── 0. Precompute covariance stats (skips if cached) ─────────────────────
run_step "Precompute covariance stats" python -m EasyEdit.easyeditor.models.rome.layer_stats \
    --model_name $MODEL \
    --layers     $ALL_LAYERS \
    --layer_tmp  "model.layers.{}.mlp.down_proj" \
    --dataset    wikipedia \
    --sample_size 100000 \
    --precision  float32 \
    --stats_dir  data/stats

# ── 1. AlphaEdit: re-edit + eval + analyses ────────────────────────────────
echo "Cleaning up GPU after precompute..."
wait_for_gpu
run_step "AlphaEdit edit + analysis" python analysis/run_analysis.py \
    --model   $MODEL \
    --methods alphaedit \
    --data-dir $DATA_DIR \
    --hparams-alphaedit EasyEdit/hparams/AlphaEdit/llama3-8b.yaml \
    --batch-size         128 \

# ── 2. analyze_tables: AlphaEdit ──────────────────────────────────────────
run_step "analyze_tables: AlphaEdit" python analysis/analyze_tables.py \
    --subsets \
        pre-edit:${DATA_DIR}/composition_incorrect \
        now_correct:${ALPHAEDIT_OUT}/now_correct \
        still_incorrect:${ALPHAEDIT_OUT}/still_incorrect \
        still_inconsistent:${ALPHAEDIT_OUT}/still_inconsistent \
        broken_hops:${ALPHAEDIT_OUT}/broken_hops

# ── 3. IFMET: re-edit + eval + analyses ───────────────────────────────────
echo "Cleaning up GPU before IFMET..."
pkill -9 -f "run_analysis.py" 2>/dev/null || true
sleep 5
wait_for_gpu
run_step "IFMET edit + analysis" python analysis/run_analysis.py \
    --model   $MODEL \
    --methods ifmet \
    --data-dir $DATA_DIR \
    --hparams-ifmet-shallow EasyEdit/hparams/IFMET/llama3-8b-shallow.yaml \
    --hparams-ifmet-deep    EasyEdit/hparams/IFMET/llama3-8b-deep.yaml \
    --batch-size         128 \

# ── 4. analyze_tables: IFMET ─────────────────────────────────────────────
run_step "analyze_tables: IFMET" python analysis/analyze_tables.py \
    --subsets \
        pre-edit:${DATA_DIR}/composition_incorrect \
        now_correct:${IFMET_OUT}/now_correct \
        still_incorrect:${IFMET_OUT}/still_incorrect \
        still_inconsistent:${IFMET_OUT}/still_inconsistent \
        broken_hops:${IFMET_OUT}/broken_hops

# ── 5. EAP-AlphaEdit: re-edit + eval + analyses ───────────────────────────
echo "Cleaning up GPU before EAP-AlphaEdit..."
pkill -9 -f "run_analysis.py" 2>/dev/null || true
sleep 5
wait_for_gpu
run_step "EAP-AlphaEdit edit + analysis" python analysis/run_analysis.py \
    --model   $MODEL \
    --methods eap_alphaedit \
    --data-dir $DATA_DIR \
    --hparams-eap-alphaedit EasyEdit/hparams/AlphaEdit_Circuit/llama3-8b.yaml \
    --batch-size         128 \
    --force

# ── 6. analyze_tables: EAP-AlphaEdit ──────────────────────────────────────
run_step "analyze_tables: EAP-AlphaEdit" python analysis/analyze_tables.py \
    --subsets \
        pre-edit:${DATA_DIR}/composition_incorrect \
        now_correct:${EAP_ALPHAEDIT_OUT}/now_correct \
        still_incorrect:${EAP_ALPHAEDIT_OUT}/still_incorrect \
        still_inconsistent:${EAP_ALPHAEDIT_OUT}/still_inconsistent \
        broken_hops:${EAP_ALPHAEDIT_OUT}/broken_hops

# ── 7. Revised_AlphaEdit: re-edit + eval + analyses ───────────────────────
echo "Cleaning up GPU before Revised_AlphaEdit..."
pkill -9 -f "run_analysis.py" 2>/dev/null || true
sleep 5
wait_for_gpu
run_step "Revised_AlphaEdit edit + analysis" python analysis/run_analysis.py \
    --model   $MODEL \
    --methods revised_alphaedit \
    --data-dir $DATA_DIR \
    --hparams-revised-alphaedit EasyEdit/hparams/Revised_AlphaEdit/llama3-8b.yaml \
    --batch-size         128 \
    --force

# ── 8. analyze_tables: Revised_AlphaEdit ──────────────────────────────────
run_step "analyze_tables: Revised_AlphaEdit" python analysis/analyze_tables.py \
    --subsets \
        pre-edit:${DATA_DIR}/composition_incorrect \
        now_correct:${REVISED_ALPHAEDIT_OUT}/now_correct \
        still_incorrect:${REVISED_ALPHAEDIT_OUT}/still_incorrect \
        still_inconsistent:${REVISED_ALPHAEDIT_OUT}/still_inconsistent \
        broken_hops:${REVISED_ALPHAEDIT_OUT}/broken_hops

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
