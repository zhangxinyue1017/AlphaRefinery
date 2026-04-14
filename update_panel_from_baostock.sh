#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_INPUT_ROOT="${ALPHAREFINERY_BAOSTOCK_DAILY_ROOT:-/root/dmd/BaoStock/daily}"
DEFAULT_OUTPUT="${ALPHAREFINERY_PANEL_PATH:-/root/dmd/BaoStock/panel.parquet}"
DEFAULT_PYTHON="${PYTHON_BIN:-python}"

INPUT_ROOT="${DEFAULT_INPUT_ROOT}"
OUTPUT_PATH="${DEFAULT_OUTPUT}"
START_DATE=""
END_DATE=""
DUPLICATE_POLICY="error"
PYTHON_BIN="${DEFAULT_PYTHON}"

usage() {
  cat <<EOF
Usage: ./update_panel_from_baostock.sh [options]

Rebuild a standard panel parquet from BaoStock daily CSV files.

Options:
  --input-root PATH         BaoStock daily root
                            default: ${DEFAULT_INPUT_ROOT}
  --output PATH             panel output path
                            default: ${DEFAULT_OUTPUT}
  --start-date YYYY-MM-DD   optional inclusive start date
  --end-date YYYY-MM-DD     optional inclusive end date
  --duplicate-policy VALUE  duplicate handling: error|first|last|mean
                            default: error
  --python PYTHON           python executable
                            default: ${DEFAULT_PYTHON}
  -h, --help                show this help

Environment overrides:
  ALPHAREFINERY_BAOSTOCK_DAILY_ROOT
  ALPHAREFINERY_PANEL_PATH
  PYTHON_BIN
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-root)
      INPUT_ROOT="$2"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --start-date)
      START_DATE="$2"
      shift 2
      ;;
    --end-date)
      END_DATE="$2"
      shift 2
      ;;
    --duplicate-policy)
      DUPLICATE_POLICY="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${DUPLICATE_POLICY}" in
  error|first|last|mean)
    ;;
  *)
    echo "Invalid --duplicate-policy: ${DUPLICATE_POLICY}" >&2
    exit 1
    ;;
esac

if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "Input root does not exist: ${INPUT_ROOT}" >&2
  exit 1
fi

OUTPUT_DIR="$(dirname "${OUTPUT_PATH}")"
OUTPUT_NAME="$(basename "${OUTPUT_PATH}")"
REPORT_PATH="${OUTPUT_PATH}.baostock_report.json"
OUTPUT_EXT=""
OUTPUT_STEM="${OUTPUT_NAME}"

if [[ "${OUTPUT_NAME}" == *.* ]]; then
  OUTPUT_EXT=".${OUTPUT_NAME##*.}"
  OUTPUT_STEM="${OUTPUT_NAME%.*}"
fi

mkdir -p "${OUTPUT_DIR}"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
TMP_OUTPUT="${OUTPUT_DIR}/.${OUTPUT_STEM}.${STAMP}.tmp${OUTPUT_EXT}"
TMP_REPORT="${TMP_OUTPUT}.baostock_report.json"

cleanup() {
  rm -f "${TMP_OUTPUT}" "${TMP_REPORT}"
}
trap cleanup EXIT

cd "${ROOT_DIR}"

CMD=(
  "${PYTHON_BIN}" -m factors_store._vendor.gpqlib_runtime.data.data_process.baostock
  --input-root "${INPUT_ROOT}"
  --output "${TMP_OUTPUT}"
  --duplicate-policy "${DUPLICATE_POLICY}"
  --report-path "${TMP_REPORT}"
)

if [[ -n "${START_DATE}" ]]; then
  CMD+=(--start-date "${START_DATE}")
fi
if [[ -n "${END_DATE}" ]]; then
  CMD+=(--end-date "${END_DATE}")
fi

echo "[panel] rebuilding from ${INPUT_ROOT}"
echo "[panel] temp output: ${TMP_OUTPUT}"
"${CMD[@]}"

mv -f "${TMP_OUTPUT}" "${OUTPUT_PATH}"
mv -f "${TMP_REPORT}" "${REPORT_PATH}"

trap - EXIT

echo "[panel] updated: ${OUTPUT_PATH}"
echo "[panel] report: ${REPORT_PATH}"
