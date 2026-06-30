# EPDP EP权重分片修复总结

## 问题诊断

### 现象
启动`vllm serve --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel --enable-dp-attention`时OOM：
- 每个GPU分配~135GB（接近完整模型）
- 日志显示加载全部64个safetensors分片
- 预期：每个GPU只加载32/256个专家（~40-50GB）

### 根因
`FusedMoEParallelConfig.make()`中的EP size计算逻辑不支持enable_dp_attention模式：

**原代码（错误）**:
```python
ep_size = tp_size  # TP1 → ep_size=1 → 加载全部256个专家
```

**预期行为（DP-attention模式）**:
```python
ep_size = dp_size * tp_size  # DP8*TP1 → ep_size=8 → 每个rank加载32个专家
```

## 修复内容

### 文件：vllm/model_executor/layers/fused_moe/config.py

**修改位置**: Line 1216-1235（`FusedMoEParallelConfig.make()`方法）

**修复逻辑**:
```python
# Check for enable_dp_attention mode (DP-attention + EP experts)
enable_dp_attention = getattr(vllm_parallel_config, "enable_dp_attention", False)
if enable_dp_attention:
    # DP-attention mode: fold DP dimension into EP topology
    # ep_size = dp_size * tp_size, attention stays in DP mode
    ep_size = dp_size * tp_size
    ep_rank = dp_rank * tp_size + tp_rank
else:
    # Standard EP mode
    ep_size = tp_size
    ep_rank = tp_rank
```

### 关键点
1. **enable_dp_attention模式**: ep_size = dp_size * tp_size = 8 * 1 = 8
2. **标准EP模式**: ep_size = tp_size（保持原有逻辑）
3. **EP rank计算**: ep_rank = dp_rank * tp_size + tp_rank（拓扑折叠）

## 完整的EPDP支持改动

| 文件 | 修改内容 | 目的 |
|-----|---------|------|
| vllm/config/parallel.py | 添加`enable_dp_attention: bool`字段 | 定义配置项 |
| vllm/engine/arg_utils.py | 添加`--enable-dp-attention` CLI参数 | 用户接口 |
| vllm/models/deepseek_v4/amd/plugin/config.py | Plugin读取enable_dp_attention | ATOM集成 |
| **vllm/model_executor/layers/fused_moe/config.py** | **FusedMoEParallelConfig支持DP-attention** | **EP权重分片** |

## 验证方法

### 1. 重新构建镜像
```bash
cd /home/qilihuan/dsv4-pro-dev/vllm
docker build -f Dockerfile.epdp-simple -t qilihuan/vllm:epdp-1.1 .
```

### 2. 启动测试
```bash
docker run -d --name test-epdp-fix \
  --device /dev/kfd --device /dev/dri \
  --ipc host --network host \
  -v /mnt:/mnt \
  qilihuan/vllm:epdp-1.1 sleep infinity

docker exec test-epdp-fix bash -c '
export MORI_GPU_ARCHS=gfx950
vllm serve /mnt/deepseek-v4-pro \
  --tensor-parallel-size 1 --data-parallel-size 8 \
  --enable-expert-parallel --enable-dp-attention \
  --gpu-memory-utilization 0.8 \
  ... &
'
```

### 3. 检查权重加载
监控日志，应该看到**每个Worker只加载8个分片**（64/8），而不是全部64个：
```
(Worker_DP0_EP0) Loading safetensors shards: 8/8
(Worker_DP1_EP1) Loading safetensors shards: 8/8
...
```

### 4. 检查GPU内存
```bash
rocm-smi --showmeminfo vram | grep Used
```
每个GPU应该分配~40-50GB，而不是135GB。

## Git Commit

```bash
git log --oneline -1
ac2eb7ed3 fix: Enable EP weight sharding in DP-attention mode
```

推送到: https://github.com/qilihuan/vllm/tree/epdp-support

## 预期结果

✅ **权重正确分片**: 每个EP rank加载32/256个专家  
✅ **内存占用正常**: ~40-50GB/GPU，而非135GB  
✅ **服务启动成功**: 无OOM错误  
✅ **功能正常**: GSM8K准确率 > 0.96
