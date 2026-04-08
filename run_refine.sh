#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ROOT_DIR}/llm_refine_provider_env.sh"
ENV_TEMPLATE="${ROOT_DIR}/llm_refine_provider_env.example.sh"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}"
  echo "Create it from the tracked template first:"
  echo "  cp ${ENV_TEMPLATE} ${ENV_FILE}"
  echo "  source ${ENV_FILE}"
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

if [[ $# -lt 1 ]]; then
  echo "Usage: ./run_refine.sh <family> [extra args...]"
  echo "Example: ./run_refine.sh open_volume_correlation --n-candidates 3"
  exit 1
fi

cd "${ROOT_DIR}"

python -m factors_store.llm_refine.cli.run_refine_loop \
  --family "$1" \
  "${@:2}"
