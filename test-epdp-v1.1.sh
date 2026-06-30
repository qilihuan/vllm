#!/bin/bash
# 测试vLLM EPDP v1.1 - EP权重分片验证

set -e

IMAGE_NAME="qilihuan/vllm:epdp-1.1"
CONTAINER_NAME="test-epdp-v1.1"
EXPERIMENT_NAME="epdp_v1.1_test"

echo "=========================================="
echo "vLLM EPDP v1.1 测试"
echo "目标: 验证EP权重分片是否生效"
echo "=========================================="

# 1. 启动容器
echo ""
echo "[1/5] 启动测试容器..."
docker ps -a | grep ${CONTAINER_NAME} && docker rm -f ${CONTAINER_NAME} || true

docker run -d --name ${CONTAINER_NAME} \
  --device /dev/kfd --device /dev/dri \
  --group-add video \
  --security-opt seccomp=unconfined \
  --security-opt label=disable \
  --ipc host --network host \
  --cap-add SYS_PTRACE \
  --shm-size 64g \
  -v /mnt:/mnt \
  -v /home/qilihuan/dsv4-pro-dev:/hql-dev \
  -v /mnt/deepseek-v4-pro:/.cache/huggingface/deepseek-v4-pro:ro \
  -e HF_HOME=/mnt/huggingface-cache \
  ${IMAGE_NAME} \
  sleep infinity

echo "✅ 容器启动成功: ${CONTAINER_NAME}"

# 2. 设置环境变量并启动服务
echo ""
echo "[2/5] 启动vLLM服务（DP8+EP8）..."
docker exec ${CONTAINER_NAME} bash -c "
set -e

# 环境变量
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# MoRI配置
export MORI_GPU_ARCHS=gfx950
export MORI_SHMEM_MODE=ISOLATION
export MORI_DISPATCH_DTYPE=bf16
export MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384

# DeepSeek-V4 MXFP4精度标志
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export AITER_LOG_LEVEL=WARNING

# 清理缓存
rm -rf /root/.cache/atom/* /root/.mori/jit/gfx942_* 2>/dev/null || true

cd /hql-dev
mkdir -p runs/${EXPERIMENT_NAME}

# 启动服务
nohup vllm serve /mnt/deepseek-v4-pro \
  --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --host 0.0.0.0 --port 19090 \
  --tensor-parallel-size 1 --data-parallel-size 8 \
  --enable-expert-parallel \
  --enable-dp-attention \
  --distributed-executor-backend mp \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.8 \
  --kv-cache-dtype fp8 \
  --max-num-seqs 256 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --enforce-eager \
  > runs/${EXPERIMENT_NAME}/server.log 2>&1 &

echo 'Service started, waiting for initialization...'
"

echo "✅ 服务启动命令已执行"

# 3. 监控启动日志（关键验证点）
echo ""
echo "[3/5] 监控权重加载过程..."
echo "等待30秒..."
sleep 30

docker exec ${CONTAINER_NAME} bash -c "
# 检查权重加载分片数
echo '=== 权重加载验证 ==='
grep 'Loading safetensors shards' /hql-dev/runs/${EXPERIMENT_NAME}/server.log | tail -10

# 检查Worker名称
echo ''
echo '=== Worker验证 ==='
grep 'Worker_DP.*_EP' /hql-dev/runs/${EXPERIMENT_NAME}/server.log | head -10

# 检查是否有OOM
echo ''
echo '=== OOM检查 ==='
grep -i 'out of memory\|OOM' /hql-dev/runs/${EXPERIMENT_NAME}/server.log || echo 'No OOM errors ✅'
"

# 4. 检查GPU内存
echo ""
echo "[4/5] 检查GPU内存占用..."
docker exec ${CONTAINER_NAME} bash -c "
rocm-smi --showmeminfo vram | grep -E 'GPU\[|Used'
"

# 5. 等待服务完全启动
echo ""
echo "[5/5] 等待服务完全启动..."
echo "监控日志（Ctrl+C退出）："
echo ""
docker exec ${CONTAINER_NAME} bash -c "
timeout 300 tail -f /hql-dev/runs/${EXPERIMENT_NAME}/server.log 2>/dev/null | grep -m 1 'Application startup complete' || true
"

echo ""
echo "=========================================="
echo "服务状态检查："
echo ""
docker exec ${CONTAINER_NAME} curl -s http://127.0.0.1:19090/v1/models 2>/dev/null && echo "✅ API可用" || echo "⚠️ API未就绪"

echo ""
echo "=========================================="
echo "后续测试步骤："
echo ""
echo "1. 进入容器："
echo "   docker exec -it ${CONTAINER_NAME} bash"
echo ""
echo "2. Smoke test："
echo "   curl -s http://127.0.0.1:19090/v1/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"model\":\"deepseek-ai/DeepSeek-V4-Pro\",\"prompt\":\"Q: 7*6=? A:\",\"max_tokens\":5,\"temperature\":0}'"
echo ""
echo "3. 查看完整日志："
echo "   tail -f /hql-dev/runs/${EXPERIMENT_NAME}/server.log"
echo "=========================================="
