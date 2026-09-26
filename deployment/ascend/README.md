# 昇腾适配预留说明

更新时间：2026-07-22

## 当前结论

本目录已从“仅有预留方案”推进到“Qwen3.5/MindSpeed-MM 上机代码已准备”。当前仍未在昇腾 NPU 上运行，尚未完成真实模型构建、100 步精度对齐或性能测试。

Qwen3.5 主线现有内容：

- `third_party/MindSpeed-MM/`：用户提供的参考快照；
- `mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py`：缺失目录的最小注册适配层；
- `scripts/check_mindspeed_qwen35_bundle.py`：静态/动态严格预检；
- `scripts/convert_qwen35_weights.sh`：Hugging Face 权重转 DCP；
- `scripts/run_mindspeed_qwen35.sh`：单 NPU 100 步运行；
- `scripts/analyze_qwen35_training_log.py`：精度与性能日志验收。

当前与本次单卡迁移对应的是纯 PyTorch 监督式基线：

- 框架：纯 PyTorch，不依赖 DGL/PyG；
- 模型文件：`model_service/model.py`；
- 训练入口：`scripts/train_baseline.py`；
- 推理入口：`scripts/predict_baseline.py`；
- NPU 运行入口：`scripts/run_ascend_baseline.sh`；
- 数据：`experiments/results/sichuan_processed.npz`；
- NPU 配置：`experiments/configs/sichuan_baseline_ascend.yaml`。

`model_service/gat_cobo/` 中另有 DGL `GATConv` 版本，仍属于独立的后续适配对象，不与本次纯 PyTorch 基线混为一谈。

## 目录文件

| 文件 | 用途 |
|---|---|
| `environment.md` | 昇腾环境、依赖和版本约束 |
| `migration_plan.md` | `torch_npu` 与 ONNX 两条迁移路线 |
| `operator_compatibility.md` | 当前模型算子兼容风险清单 |
| `benchmark_plan.md` | 后续性能测试方法和指标定义 |
| `TODO.md` | 复赛阶段需要实际完成的任务 |
| `910_ubuntu22_runbook.md` | Ascend 910/910B + Ubuntu 22.04 单卡实操命令 |
| `qwen35_official_task_runbook.md` | Qwen3.5-0.8B 官方口径 100 步上机命令 |
| `qwen35_server.env.example` | 服务器路径变量模板，不含密钥 |

## 真实性边界

- 不能把本机 CPU 推理耗时写成昇腾性能；
- 不能把接口原型 benchmark 写成模型 NPU benchmark；
- 没有真实昇腾环境前，只能写“迁移方案”和“测试计划”；
- 昇腾测试必须记录硬件型号、CANN 版本、torch_npu 版本、模型 commit、输入规模和测试命令。

## 推荐执行顺序

1. 选择与参考包兼容的官方 ModelArts NPU 镜像；
2. 准备 Qwen3.5-0.8B 权重、DCP checkpoint 和官方验证数据；
3. 执行 Qwen3.5 动态预检和 100 步训练；
4. 完成 loss/grad norm 对齐后再做单变量优化；
5. 接通 Qwen3.5 与证据约束 Skill；
6. 最后把纯 PyTorch GAT-COBO NPU 迁移作为业务增强项。
