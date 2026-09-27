# 图织天网 README

团队名称 __________    团队成员 __________

## 项目与交付范围

本说明对应 2026 年 9 月 27 日 HF 直接初始化的 Qwen3.5-0.8B 昇腾迁移实验。新版运行完成 100 步训练、检查点严格加载及文本和图片推理冒烟测试。迁移源码包不包含模型权重、数据、虚拟环境或完整业务 Demo。

业务 Demo 与训练实验是两条独立链路。现有 Demo 使用预测样例和规则报告，尚无证据证明已经接入新版 NPU 模型；命令行推理成功不代表业务链路验收。

## 镜像和实际环境

平台截图标示 HiDevLab TorchNPU 2.10.0，默认 Ubuntu 22.04、Python 3.11。这是平台镜像名称和说明，不是训练虚拟环境版本；镜像不可变 ID 或摘要未取得。

新版推理报告记录 Python 3.10.12、torch 2.7.1+cpu、torch_npu 2.7.1.post4、Transformers 5.2.0，设备 npu:0。torch 版本后缀 +cpu 不能单独作为 CPU 回退判断，推理实际使用 torch_npu 的 NPU 后端。

历史环境记录为 CANN 9.0.0，虚拟环境位于 /workspace/tuzhitianwang/.venv-qwen35-checked。新版精简证据包没有完整环境锁定清单，因此其他依赖不能一概宣称已重新逐项核验。训练封装需要 PyYAML，脚本不会自动安装依赖。

## 新版代码入口

skills/qwen35-ascend-migrate/scripts/migrate.py：输出执行计划，并在明确确认后启动训练。

同目录 run_hf_training.py：读取成功 YAML，保留 HF 初始化设置，改写模型、数据、缓存和保存路径，单卡启动 MindSpeed-MM trainer。

同目录 verify_inference.py：离线文本冒烟测试，可严格恢复可信 DCP 检查点。third_party/MindSpeed-MM 中的模型加载与 DDP 同步修复来自新版云端证据。

旧入口 scripts/run_mindspeed_qwen35.sh、旧 DCP/meta 配置和旧参考日志属于历史路径，不用于新版验收。新版封装完成本地测试，尚未以交付 ZIP 在云端重跑。

---

## 新版训练启动

以下在已有依赖的昇腾 Linux 环境运行。B 指向解压后的新版源码包根目录，不是旧包。模型目录应含 config、safetensors、tokenizer 和 processor 文件；数据目录必须同时包含标注及图片。先核对实际路径。

```bash
R=/workspace/tuzhitianwang
B="$R/qwen35_hf_source_20260927"
V="$R/.venv-qwen35-checked/bin/python"
M="$R/models/Qwen3.5-0.8B"
D="$R/data/official_coco/extracted/dataset/annotations_slim.json"
C=/usr/local/Ascend/cann/set_env.sh
S="$B/skills/qwen35-ascend-migrate/scripts"
E="$B/verified_run/20260927/qwen35_ascend_server_bundle"
E="$E/results/ascend_qwen35/runs"
CFG="$E/hf_init_full_20260927_081557/train100.yaml"
O="$R/reproduce_hf_$(date +%Y%m%d_%H%M%S)"
"$V" "$S/migrate.py" --stage train \
  --bundle "$B" --python "$V" --hf-model "$M" \
  --dataset "$D" --cann-env "$C" --output "$O" \
  --verified-config "$CFG"
```

该命令只打印计划，不启动训练。需要实际重跑时，在同一条命令末尾增加 --execute --confirm-training。已有成功证据不要求重复训练。不要用默认 preflight 代替新版训练计划；它检查旧 DCP 资产，需额外 --dcp，与本入口不同。

输出目录必须是新目录。外层保存 skill_receipt.json 与 skill_console.log；training 子目录保存 train100.yaml、train.log、training_receipt.json，检查点保存到 training/checkpoints/iter_0000100。退出码为 0 仅代表命令完成，应继续检查步数、权重与推理结果。

训练配置为 100 步、micro batch 1、梯度累积 8、seed 42、lr 1e-5、BF16 参数与 FP32 归约、重计算开启。初始化为 meta=False、load=''、load_rank0_and_broadcast=False；输出不加载旧 DCP，也不自动与旧参考日志比较。

---

## 新版检查点文本验证

沿用前页变量，在同一 shell 中设置 CK 指向自己可信的检查点。下例使用已有成功运行，不修改原权重。DCP 元数据含 pickle，不能加载来源不可信的检查点。

```bash
CK="$R/qwen35_ascend_server_bundle/results/ascend_qwen35/runs"
CK="$CK/hf_init_full_20260927_081557/checkpoints/iter_0000100"
I="$R/inference_checks/readme_$(date +%Y%m%d_%H%M%S)"
source "$C"
"$V" "$S/verify_inference.py" \
  --model "$M" --checkpoint "$CK" \
  --mindspeed-root "$B/third_party/MindSpeed-MM" \
  --output "$I" --max-new-tokens 128
```

该命令也是计划模式；实际验证时增加 --execute --trust-local-checkpoint。先检查输出目录尚不存在。成功报告要求严格加载、有限 logits 和非空生成文本；不验证多模态准确率、优化器恢复或断点续训。

## 结果与证据索引

成功运行 hf_init_full_20260927_081557：100 步，累计 800 个批次样本，末步 loss 1.414928。全程平均 29.1377 秒/步、0.2746 样本/秒；第 50–100 步平均 29.6119 秒/步。仅单次运行，没有可主张的加速比或独立精度对齐。

源码包 verified_run/20260927 下包含成功 train.log、train100.yaml，以及 inference_checks 下的文本、单图、20 图比较证据。UPDATE_MANIFEST_20260927.json 记录本次更新文件来源和哈希；根目录 MANIFEST.json 是历史包清单；同目录中旧运行文件属于历史证据。

verified_run/20260927/performance 保存 performance.json 和 steps.csv。新检查点独立备份的整包 SHA256 为 8387bbd7b3b54273964578e321fb3ed852f21212af4b770c9561c8937dd44ba8；106 分片及内部文件校验通过，但检查点不在源码 ZIP 内。

## 仓库与 PR

仓库：https://github.com/b3720816/tuzhitianwang-qwen35-ascend

PR：https://github.com/b3720816/tuzhitianwang-qwen35-ascend/pull/1

此前记录该 PR 已合并到团队自建仓库，不是上游 PR。本次 HF 更新由独立分支提交审阅，不能把 PR #1 当作新版代码的合并凭证。仓库链接也不代表评分认可。

## 仍待完成

新初始化路径的可比精度基线、业务 Demo 的 NPU 联调、新版 PR 审阅和整包最终验收尚未完成。20 图检查是 AI 辅助诊断，不替代正式精度对齐。团队信息留空，最终提交格式和签名另行核对。
