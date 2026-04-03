#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/llm_refine_provider_env.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: ./run_refine.sh <family> [extra args...]"
  echo "Example: ./run_refine.sh open_volume_correlation --n-candidates 3"
  exit 1
fi

cd "${ROOT_DIR}"

python -m factors_store.llm_refine.cli.run_refine_loop \
  --family "$1" \
  "${@:2}"
