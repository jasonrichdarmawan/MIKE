#!/usr/bin/env bash
# Run neuron attribution and terminate the Lambda Cloud instance on completion.
#
# Usage (on the Lambda instance):
#   export LAMBDA_API_KEY=<your-api-key>
#   export LAMBDA_INSTANCE_ID=<your-instance-id>
#   bash run_and_terminate.sh [gpt2|llama3]
#
# LAMBDA_API_KEY    : from https://cloud.lambdalabs.com/api-keys
# LAMBDA_INSTANCE_ID: from https://cloud.lambdalabs.com/instances (or the dashboard URL)
#
# The instance is terminated via trap, so it runs even if the script errors or is killed.

set -euo pipefail

MODEL=${1:-gpt2}

# ── Validate env vars ────────────────────────────────────────────────────────

if [[ -z "${LAMBDA_API_KEY:-}" ]]; then
    echo "ERROR: LAMBDA_API_KEY is not set." >&2
    exit 1
fi

if [[ -z "${LAMBDA_INSTANCE_ID:-}" ]]; then
    echo "ERROR: LAMBDA_INSTANCE_ID is not set." >&2
    exit 1
fi

# ── Terminate on exit (success, error, or kill) ──────────────────────────────

terminate_instance() {
    echo ""
    echo "=== Terminating Lambda instance ${LAMBDA_INSTANCE_ID} ==="
    curl -s -X POST \
        "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate" \
        -H "Authorization: Bearer ${LAMBDA_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{\"instance_ids\": [\"${LAMBDA_INSTANCE_ID}\"]}"
    echo ""
}

trap terminate_instance EXIT

# ── Select model config ───────────────────────────────────────────────────────

case "${MODEL}" in
    gpt2)
        MODEL_NAME="gpt2"
        DATA_DIR=~/CaKE/datasets/gpt2
        LOG=~/neuron_attribution_gpt2.log
        ;;
    llama3)
        MODEL_NAME="meta-llama/Meta-Llama-3-8B"
        DATA_DIR=~/CaKE/datasets/Meta-Llama-3-8B
        LOG=~/neuron_attribution_llama3.log
        ;;
    *)
        echo "ERROR: Unknown model '${MODEL}'. Use 'gpt2' or 'llama3'." >&2
        exit 1
        ;;
esac

# ── Run attribution ───────────────────────────────────────────────────────────

echo "=== Starting neuron attribution: model=${MODEL_NAME} ==="
echo "    Log: ${LOG}"
echo ""

cd ~/CaKE/analysis

python3 neuron_attribution.py \
    --model-name "${MODEL_NAME}" \
    --data-dir   "${DATA_DIR}" \
    --subsets correct incorrect inconsistent \
    --prompts two-hop first-hop second-hop \
    2>&1 | tee "${LOG}"
