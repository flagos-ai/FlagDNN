#!/usr/bin/env bash

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; NVIDIA driver is not configured" >&2
  exit 1
fi

gpu_count="$(
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l
)"
if [[ "${gpu_count}" -eq 0 ]]; then
  echo "No NVIDIA GPU is visible" >&2
  exit 1
fi

echo "Visible NVIDIA GPUs: ${gpu_count}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv
