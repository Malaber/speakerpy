#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv .venv
  else
    python3 -m venv .venv
  fi
fi
if ! .venv/bin/python -m invoke --version >/dev/null 2>&1; then
  .venv/bin/python -m pip install 'invoke>=2.2,<3'
fi
if [[ ! -f .venv/.speakerpy-ready ]]; then
  .venv/bin/python -m invoke setup
fi
exec .venv/bin/python -m invoke run --browser
