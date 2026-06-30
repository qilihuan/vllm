#!/bin/bash
# 构建vLLM EPDP v1.1 (with EP weight sharding fix)

set -e

IMAGE_NAME="qilihuan/vllm:epdp-1.1"
DOCKERFILE="Dockerfile.epdp"
BUILD_LOG="build-epdp-v1.1.log"

echo "=========================================="
echo "构建vLLM EPDP v1.1镜像"
echo "基础镜像: sabreshao/vllm:aiter_0620_full"
echo "vLLM分支: qilihuan/vllm@epdp-support"
echo "关键修复: EP weight sharding (FusedMoEParallelConfig)"
echo "目标镜像: ${IMAGE_NAME}"
echo "=========================================="

# 检查Dockerfile
if [ ! -f "${DOCKERFILE}" ]; then
    echo "错误: ${DOCKERFILE} 不存在"
    exit 1
fi

# 显示最新的3个commit
echo ""
echo "vLLM最新提交："
git log --oneline -3

echo ""
echo "开始构建..."
docker build \
    -f ${DOCKERFILE} \
    -t ${IMAGE_NAME} \
    . 2>&1 | tee ${BUILD_LOG}

# 检查构建结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 构建成功！"
    echo "镜像: ${IMAGE_NAME}"
    echo "=========================================="

    # 显示镜像信息
    docker images ${IMAGE_NAME}

    echo ""
    echo "验证镜像内容..."
    docker run --rm ${IMAGE_NAME} bash -c '
        echo "=== vLLM ==="
        python3 -c "from vllm import __version__; print(f\"版本: {__version__}\")"

        echo ""
        echo "=== Git信息 ==="
        cd /app/vllm
        git branch
        git log --oneline -1

        echo ""
        echo "=== 关键修复验证 ==="
        grep -A 5 "enable_dp_attention = getattr" /app/vllm/vllm/model_executor/layers/fused_moe/config.py | head -8

        echo ""
        echo "=== ParallelConfig ==="
        python3 -c "from vllm.config import ParallelConfig; print(f\"enable_dp_attention字段: {hasattr(ParallelConfig, \"enable_dp_attention\")}\")"
    '

    echo ""
    echo "=========================================="
    echo "下一步测试："
    echo ""
    echo "1. 启动容器："
    echo "   docker run -d --name test-epdp-v1.1 \\"
    echo "     --device /dev/kfd --device /dev/dri \\"
    echo "     --ipc host --network host --shm-size 64g \\"
    echo "     -v /mnt:/mnt -v /home/qilihuan/dsv4-pro-dev:/hql-dev \\"
    echo "     ${IMAGE_NAME} sleep infinity"
    echo ""
    echo "2. 查看测试说明："
    echo "   cat /home/qilihuan/dsv4-pro-dev/vllm/EPDP_FIX_SUMMARY.md"
    echo "=========================================="
else
    echo ""
    echo "❌ 构建失败，查看日志: ${BUILD_LOG}"
    exit 1
fi
