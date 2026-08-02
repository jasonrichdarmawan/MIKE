#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<EOF
Usage:
  $0 <MAIN_FOLDER_PATH> [OPTIONS]

Options:
  --comparison-metrics-path <PATH>
      Root comparison folder. Uses <PATH>/<lang>/metrics.json.

  --res-path <PATH>
      Result-file path relative to each language folder.
      Default: res_per_method.json

Examples:
  $0 outputs/ours

  $0 outputs/ours \
      --res-path custom_res_per_method.json

  $0 outputs/ours \
      --comparison-metrics-path outputs/alphaedit \
      --res-path results/res_per_method.json
EOF
    exit 1
}

[[ $# -ge 1 ]] || usage

MAIN_FOLDER_PATH="$1"
shift

COMPARISON_MAIN_FOLDER_PATH=""
RES_PATH_RELATIVE="res_per_method.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --comparison-metrics-path)
            [[ $# -ge 2 ]] || {
                echo "Error: --comparison-metrics-path requires a path." >&2
                usage
            }
            COMPARISON_MAIN_FOLDER_PATH="$2"
            shift 2
            ;;
        --res-path)
            [[ $# -ge 2 ]] || {
                echo "Error: --res-path requires a path." >&2
                usage
            }
            RES_PATH_RELATIVE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage
            ;;
    esac
done

for lang in en de fr ja zh; do
    FOLDER_PATH="${MAIN_FOLDER_PATH}/${lang}"
    METRICS_PATH="${FOLDER_PATH}/metrics.json"
    RES_PATH="${FOLDER_PATH}/${RES_PATH_RELATIVE}"

    if [[ ! -f "${METRICS_PATH}" ]]; then
        echo "Skipping ${lang}: missing metrics file:" >&2
        echo "  metrics: ${METRICS_PATH}" >&2
        continue
    fi

    cmd=(
        python output_analysis/metrics_to_statistics.py
        --metrics-path "${METRICS_PATH}"
        --res-path "${RES_PATH}"
    )

    if [[ -n "${COMPARISON_MAIN_FOLDER_PATH}" ]]; then
        COMPARISON_METRICS_PATH="${COMPARISON_MAIN_FOLDER_PATH}/${lang}/metrics.json"

        if [[ ! -f "${COMPARISON_METRICS_PATH}" ]]; then
            echo "Skipping ${lang}: comparison metrics not found:" >&2
            echo "  ${COMPARISON_METRICS_PATH}" >&2
            continue
        fi

        cmd+=(
            --comparison-metrics-path "${COMPARISON_METRICS_PATH}"
        )
    else
        cmd+=(--do-individual-method)
    fi

    echo "Processing ${lang}..."
    "${cmd[@]}"
done