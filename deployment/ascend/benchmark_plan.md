# 性能测试计划

更新时间：2026-07-22

## 1. 测试口径

性能测试分三类，材料中必须区分：

| 类型 | 含义 | 能否写成昇腾性能 |
|---|---|---|
| 接口原型测试 | 测 FastAPI 或 adapter 调用耗时 | 不能 |
| CPU 模型基线 | 测 CPU 上 GAT-COBO 推理耗时 | 不能 |
| 千问 API 测试 | 测公网接口和报告链路 | 不能 |
| Qwen3.5-0.8B 原生 NPU 训练基线 | MindSpeed-MM 官方口径 100 步训练 | 可以，必须附环境、配置和原始日志 |
| Qwen3.5-0.8B 优化后测试 | 同一环境和官方口径下启用单项优化 | 可以，需同时报告 loss/grad norm 精度 |
| GAT-COBO NPU 测试 | 可选图模型迁移测试 | 可以，但不能替代赛题要求的 Qwen3.5 训练迁移 |

## 2. 官方赛题必测指标

- 训练总步数：100；
- 精度：逐步 loss、grad norm；
- 对齐门槛：loss 和 grad norm 相对绝对误差均小于 2%；
- 性能：第 50 至 100 步 elapsed time per iteration；
- 训练配置、框架 commit、模型 revision、硬件与软件环境。

不修改框架日志、loss 计算配置或 global batch size。业务 API 指标必须与官方训练指标分表记录。

## 3. 业务链路可选指标

- 平均时延；
- P50 时延；
- P95 时延；
- 吞吐量；
- 成功率；
- 并发数；
- 测试次数；
- 测试环境；
- 输入规模；
- 是否包含报告生成。

业务研判链路可以记录：

- 首响应时延或首 Token 时延；
- 输出 Token 数和生成吞吐；
- 工具调用成功率；
- 证据一致性通过率；
- 越界生成率；
- 规则模板降级次数。

## 4. 当前 benchmark 脚本

新增脚本：

```bash
python scripts/benchmark.py --help
```

本机可做 adapter 原型测试：

```bash
HOME=<PROJECT_ROOT>/.home DGLBACKEND=pytorch \
python scripts/benchmark.py \
  --target real-adapter \
  --user-id REAL_HIGH \
  --requests 20 \
  --concurrency 2 \
  --output results/stage6_real_adapter_benchmark.json
```

该结果只能说明 Python adapter 调用耗时，不是模型实时推理耗时，也不是昇腾性能。

后端启动后可测接口：

```bash
python scripts/benchmark.py \
  --target http \
  --url http://127.0.0.1:8000/api/real-users/REAL_HIGH/risk \
  --requests 50 \
  --concurrency 5 \
  --output results/stage6_http_benchmark.json
```

## 5. Qwen3.5-0.8B 训练优化矩阵

先按官方训练配置比较：

1. 原生 NPU 100 步基线；
2. 单项框架/通信/算子优化；
3. 组合优化；
4. 每一版 loss/grad norm 精度回归；
5. 每一版第 50 至 100 步性能统计。

业务研判另固定低、中、高风险各一组输入，测试证据一致性和降级率。每次只改变计划中的变量，保存完整配置和原始输出。

## 6. 昇腾测试要求

正式昇腾性能报告必须包含：

- 硬件型号；
- CANN 版本；
- PyTorch 和 `torch_npu` 版本；
- DGL 或替代实现版本；
- 模型 commit 或文件 hash；
- 千问 3.5 checkpoint 和 MindSpeed-MM 版本/commit；
- 数据集与输入规模；
- benchmark 命令；
- 原始 JSON 输出；
- 优化前后对比；
- 精度是否保持。
