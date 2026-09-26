# 昇腾迁移计划

更新时间：2026-07-22

## 1. 迁移目标与优先级

本项目按官方 Qwen3.5 模型迁移任务执行。迁移优先级确定为：

1. P0：在真实昇腾 NPU 上把 `Qwen/Qwen3.5-0.8B` 训练流程迁移到 MindSpeed-MM；
2. P0：按官方配置运行 100 步，完成 loss 与 grad norm 精度对齐；
3. P0：统计第 50 至 100 步单步耗时，建立可复现性能基线；
4. P1：将验收合格的 Qwen3.5 与图检测证据链结合；
5. P2：在条件允许时评估 GAT-COBO 图模型迁移。

当前已归档参考快照，完成 Qwen3.5 FSDP 注册适配、配置生成、权重转换、严格预检、单卡运行和日志验收代码；同时已有纯 PyTorch 图基线 NPU 适配代码和千问 API Demo Provider。所有 Qwen3.5/MindSpeed-MM 代码仍只完成本地静态验证，没有昇腾实测结果。

## 2. 官方训练验收链路

```text
官方参考样例与版本
-> Qwen3.5-0.8B 权重和 modeling_qwen3_5.py
-> mindspeed_mm/fsdp/models/qwen3_5 适配
-> 官方配置 100 步训练
-> loss / grad norm 误差校验
-> 第 50-100 步性能统计
-> 保存环境、配置、日志和报告
```

禁止用千问在线 API 的响应时延替代上述 NPU 训练性能。

## 3. 目标业务链路

```text
用户选择研判对象
-> Qwen3.5 Skill 调用 assess_node_risk
-> 图检测 adapter 返回 RiskAssessmentResult
-> Qwen3.5 Skill 调用 trace_fraud_group
-> 证据校验器限制生成依据
-> 输出证据卡与研判报告
-> 前端展示并记录端到端耗时
```

Qwen3.5 不负责替代图模型计算风险分数；它负责工具编排、证据组织和报告表达。这样能够保留图异常检测核心能力，并让赛题要求落到真实 Skill 调用链上。

## 4. 路线 A：Qwen3.5-0.8B + MindSpeed-MM/torch_npu（主路线）

实施步骤：

1. 使用已归档的 MindSpeed-MM 参考快照及其 SHA-256 溯源记录；
2. 确认 Ascend 910/910B、CANN、PyTorch、torch_npu 与 MindSpeed-MM 的配套版本；
3. 从 ModelScope 下载 `Qwen/Qwen3.5-0.8B`，记录 revision 和许可证；
4. 使用已完成的 Transformers 5.2.0 Qwen3.5 注册适配层，并在 NPU 上验证其模型构建和前后向；
5. 使用官方 `examples/qwen3_5/qwen3_5_0.8B_config.yaml` 口径配置权重和数据路径；
6. 不改框架日志、loss 计算配置与 global batch size，运行 100 步并保存完整日志；
7. 逐步对齐官方 loss 与 grad norm，二者相对绝对误差均须小于 2%；
8. 统计第 50 至 100 步单步耗时，建立原生 NPU 基线后再逐项优化；
9. 验收通过后再增加本项目业务 Provider/Skill 接口，避免业务改造干扰官方指标；
10. 将真实结果写入性能报告，保留命令、配置、JSON 和日志。

官方精度日志显示参考配置为单卡、BF16、FSDP、micro batch size 1、gradient accumulation 8、global batch size 8、train iters 100。最终仍以下载到的官方样例和群通知为准，不能预先承诺量化、OM 或某个融合算子一定可用。

## 5. 优化方向

先满足官方训练精度口径，再按单变量方式测试；不得以改变 global batch size 或 loss 计算换取不可比的性能：

- 模型常驻和预热，避免重复加载；
- 结构化提示词压缩，减少无关上下文；
- 批处理或动态批处理；
- 图检测结果缓存和只读证据复用；
- 混合精度；
- 官方版本支持的量化、融合算子或图编译；
- Skill 工具调用与报告生成阶段的耗时拆分。

每项优化必须记录优化前后配置、输出一致性、平均/P50/P95 时延、吞吐量、峰值内存和成功率。不得用“预期提速”替代实测结果。

## 6. 路线 B：GAT-COBO 图模型迁移（可选扩展）

当前 GAT-COBO 依赖 DGL `GATConv`，迁移风险较高。只有完成赛题主线后再评估：

1. 验证 DGL 与目标 PyTorch/torch_npu 版本兼容性；
2. 检查图算子是否真实运行在 NPU，避免静默回退 CPU；
3. 若 DGL 不兼容，评估原生张量重写或其他支持路径；
4. 在相同权重和测试集上比较 CPU/NPU 输出与分类指标。

仅导出 feature branch 或 MLP 分支不等同完整 GAT-COBO 迁移。

## 7. 验收门槛

迁移完成必须同时满足：

- 提供 NPU 型号、CANN、torch、torch_npu、MindSpeed-MM 和模型版本；
- Qwen3.5-0.8B 在 MindSpeed-MM 上完成 100 步训练；
- loss 与 grad norm 相对绝对误差均小于 2%；
- 性能数据来自第 50 至 100 步原始日志；
- 业务阶段能完成至少一次受证据约束的研判闭环；
- 输出只引用结构化证据，低、中、高风险样例无越界结论；
- 提供可复现启动命令、配置和原始日志；
- 提供优化前后同口径性能数据；
- 若采用低精度或量化，报告输出质量和证据一致性不能出现不可接受退化。
