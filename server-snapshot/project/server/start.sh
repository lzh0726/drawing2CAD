#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# CUDA_VISIBLE_DEVICES should be assigned by the scheduler or set to the GPU allocated to you.
umask 077
exec .venv-server/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8001 --workers 1
