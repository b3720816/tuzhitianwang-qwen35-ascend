# 昇腾环境与依赖

更新时间：2026-07-22

## 1. 当前本机基线环境

| 项 | 当前值 |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.2.1 |
| DGL | 2.1.0 |
| torchdata | 0.7.1 |
| 设备 | CPU |
| 后端环境变量 | `HOME=<PROJECT_ROOT>/.home`、`DGLBACKEND=pytorch` |

当前本机环境只用于 CPU 基线和接口原型测试，不代表昇腾性能。

## 2. 目标硬件与系统

本次纯 PyTorch 图基线目标为：

| 项 | 目标值 |
|---|---|
| 硬件型号 | Ascend 910 或 910B，实际 SoC 以 `npu-smi info` 为准 |
| NPU 卡数 | 1 |
| CPU / 内存 | 8-16 核 / 约 32 GB |
| 操作系统 | Ubuntu 22.04 LTS，以官方镜像支持为准 |
| CANN | Qwen3.5 专项指南推荐 8.5.2；以赛题参考镜像实际版本为准 |
| Python | Qwen3.5 主线按参考包推荐使用 3.10；图基线本地脚本支持 3.11 |
| PyTorch / torch_npu | Qwen3.5 专项指南推荐 2.7.1 / 2.7.1，必须与 CANN 配套 |
| MindSpeed-MM 代码 | 已归档用户提供快照，压缩包 SHA-256 见 `third_party/MindSpeed-MM.PROVENANCE.json` |
| 赛题 checkpoint | `Qwen/Qwen3.5-0.8B`，记录 ModelScope revision |
| Transformers 结构参考 | `5.2.0` 的 `modeling_qwen3_5.py` |
| 模型权重许可证 | 待核对 |
| DGL 是否支持目标环境 | 待验证 |

纯 PyTorch 图基线安装、冒烟、训练、推理和 benchmark 命令见 `deployment/ascend/910_ubuntu22_runbook.md`；正式 Qwen 赛题流程见 `deployment/ascend/qwen35_official_task_runbook.md`。

## 3. Qwen3.5-0.8B 主线依赖风险

赛题主线优先验证 Qwen3.5-0.8B 的 MindSpeed-MM 100 步训练，而不是先迁移图模型。主要风险包括：

- 官方样例、MindSpeed-MM 仓库和本地环境 commit 是否一致；
- CANN、PyTorch、torch_npu 和 MindSpeed-MM 的版本组合是否受官方支持；
- 参考包的 `fsdp/models/qwen3_5` 原本为空；当前已加入 Transformers 5.2.0 模型类注册适配层，但尚待真实 NPU 验证；
- 量化、融合算子、图编译等能力是否支持所选 checkpoint；
- Skill 工具调用格式是否能稳定约束输出，不产生脱离证据的结论。

## 4. 图模型可选迁移风险

当前 GAT-COBO 直接依赖 DGL `GATConv`。昇腾迁移风险主要集中在：

- DGL 是否能在目标昇腾环境安装并与目标 PyTorch/`torch_npu` 版本兼容；
- DGL 图算子能否真正落到 NPU，而不是回退 CPU；
- `dgl.DGLGraph`、稀疏邻接和注意力聚合是否支持导出或迁移；
- 如果图算子不支持，需要重写 GAT 层或拆分模型。

## 5. 环境验证命令

Qwen3.5 主线在昇腾环境中先运行：

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch_npu; print(torch_npu.__version__); import torch; print(torch.npu.is_available())"
python -c "import mindspeed; print('mindspeed import ok')"
python -c "import transformers; print(transformers.__version__)"
```

随后按 `deployment/ascend/qwen35_official_task_runbook.md` 执行权重转换、严格预检和 100 步训练。当前仓库已经提供可执行入口，但尚未在真实 NPU 上运行。

图模型可选迁移再运行项目基线检查：

```bash
HOME=<PROJECT_ROOT>/.home DGLBACKEND=pytorch python scripts/check_torch_dgl.py
HOME=<PROJECT_ROOT>/.home DGLBACKEND=pytorch python scripts/predict_gat_cobo.py
```

如果 `scripts/predict_gat_cobo.py` 在 NPU 环境中运行成功，还需要补充精度对齐和性能测试，不能只看命令是否退出。
