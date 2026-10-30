#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uname -sr
python3 --version
df -h .
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
if ! command -v docker >/dev/null; then
  echo "Docker missing. Ask the administrator for an approved container runtime; no host-code fallback."
  exit 1
fi
docker info --format 'Docker server: {{.ServerVersion}}'
