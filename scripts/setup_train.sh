#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
uv_bin="${UV_BIN:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PWD/.tools/python}"

if [[ ! -x .venv-train/bin/python ]]; then
  "$uv_bin" python install --no-bin 3.12
  "$uv_bin" venv --managed-python --python 3.12 .venv-train
fi
train_python=.venv-train/bin/python
"$uv_bin" pip install --python "$train_python" --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.9.0+cu128' 'torchvision==0.24.0+cu128'

flash_wheel='flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
flash_sha='4e2f9e39313266b1544b68138b15b91ee6221eccf14f7902b7c6620351340810'
downloads="$PWD/third_party/downloads"
mkdir -p -- "$downloads"
flash_path="$downloads/$flash_wheel"
if [[ ! -f "$flash_path" ]]; then
  curl --fail --location --retry 2 --connect-timeout 10 --silent --show-error \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/$flash_wheel" \
    -o "$flash_path.part"
  printf '%s  %s\n' "$flash_sha" "$flash_path.part" | sha256sum -c -
  mv -- "$flash_path.part" "$flash_path"
fi
printf '%s  %s\n' "$flash_sha" "$flash_path" | sha256sum -c -
"$uv_bin" pip install --python "$train_python" -r requirements-model.txt \
  'numpy==1.26.4' h5py 'imageio==2.34.2' 'imageio-ffmpeg==0.6.0' ftfy wandb \
  "$flash_path" -e .
"$uv_bin" pip check --python "$train_python"

upstream="$PWD/third_party/LingBot-VA"
revision='7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb'
if [[ ! -d "$upstream/.git" ]]; then
  git clone https://github.com/Robbyant/lingbot-va.git "$upstream"
  git -C "$upstream" checkout --detach "$revision"
fi
if [[ "$(git -C "$upstream" rev-parse HEAD)" != "$revision" ]]; then
  printf 'Expected LingBot-VA at %s in %s.\n' "$revision" "$upstream" >&2
  exit 1
fi
patch="$PWD/scripts/lingbot_lazy_lerobot.patch"
if ! git -C "$upstream" apply --reverse --check "$patch" 2>/dev/null; then
  git -C "$upstream" apply "$patch"
fi
