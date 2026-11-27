#!/usr/bin/env bash
set -euo pipefail

archive_override="${ROBOTWIN_ASSET_ARCHIVE:-}"
if [[ -n "$archive_override" ]]; then
  archive_override="$(realpath "$archive_override")"
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

uv_bin="${UV_BIN:-uv}"
python_bin="${PYTHON_BIN:-python3.10}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
export UV_PYTHON_DOWNLOADS=never

assets_dir="${ROBOTWIN_ROOT:-$PWD/third_party/RoboTwin}/assets"
if [[ ! -d "$assets_dir/embodiments" ]]; then
  archive="${archive_override:-$PWD/third_party/downloads/embodiments.zip}"
  if [[ ! -f "$archive" ]]; then
    printf 'RoboTwin archive missing at %s. See LOCAL_SETUP.md.\n' "$archive" >&2
    exit 1
  fi
  printf '%s  %s\n' '85ffeff55a5066def5931224a85cfa3f8abaa1fbf779bd17789a7fb3f85bc789' "$archive" | sha256sum -c -
  mkdir -p -- "$assets_dir"
  staging_dir="$(mktemp -d "$assets_dir/.embodiments.XXXXXX")"
  trap 'rm -rf -- "$staging_dir"' EXIT
  unzip -q "$archive" 'embodiments/*' -d "$staging_dir"
  mv -T -- "$staging_dir/embodiments" "$assets_dir/embodiments"
fi

for robot in aloha-agilex franka-panda ur5-wsg; do
  if [[ ! -d "$assets_dir/embodiments/$robot" ]]; then
    printf 'RoboTwin robot assets missing at %s/embodiments/%s\n' "$assets_dir" "$robot" >&2
    exit 1
  fi
done

if [[ ! -x .venv/bin/python ]]; then
  "$uv_bin" venv --python "$python_bin" .venv
fi
"$uv_bin" pip install --python .venv/bin/python -r requirements-sim.txt -e . pytest
