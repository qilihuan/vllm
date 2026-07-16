#!/bin/bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-glm52-fp8-tp8-quickreduce-int4:final-gemm}"

docker build \
  -t "${IMAGE_TAG}" \
  -f docker/context/glm52_fp8_tp8_quickreduce_int4/Dockerfile \
  docker/context/glm52_fp8_tp8_quickreduce_int4

echo "Built image: ${IMAGE_TAG}"
