#!/usr/bin/env bash
set -euo pipefail

# Portable validation runner.
# Run from anywhere inside the cloned repository.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VALIDATION_SCRIPT="${VALIDATION_SCRIPT:-$PROJECT_DIR/validation/run_validation_full.py}"
LOG_FILE="${VALIDATION_LOG:-$PROJECT_DIR/validation/validation_log.txt}"

if [[ ! -f "$VALIDATION_SCRIPT" ]]; then
  echo "Validation script not found: $VALIDATION_SCRIPT" >&2
  echo "Set VALIDATION_SCRIPT to the path of the validation entrypoint." >&2
  exit 1
fi

cd "$PROJECT_DIR"
"$PYTHON_BIN" "$VALIDATION_SCRIPT" > "$LOG_FILE" 2>&1
echo "DONE" >> "$LOG_FILE"
echo "Validation log written to: $LOG_FILE"
