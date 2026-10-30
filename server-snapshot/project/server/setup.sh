#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash server/preflight.sh
python3 -m venv .venv-server
.venv-server/bin/python -m pip install --upgrade pip
# Candidate CUDA build. Verify with your actual driver; this script never changes the driver.
.venv-server/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
.venv-server/bin/python -m pip install -r server/requirements.txt
.venv-server/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
docker build -f server/Dockerfile.cad -t drawing2cad-executor:0.1 .
.venv-server/bin/python -m pip freeze > server/requirements.linux.lock.txt
if [ ! -e server/server.env ]; then
  .venv-server/bin/python -c 'import secrets,pathlib; p=pathlib.Path("server/server.env"); p.write_text("CAD_MODEL=Qwen/Qwen3-VL-8B-Instruct\nCAD_MODEL_REVISION=main\nCAD_SERVER_TOKEN="+secrets.token_urlsafe(32)+"\n",encoding="utf-8"); p.chmod(0o600)'
fi
echo "Setup complete. Secret stored in server/server.env; never paste it into chat."
