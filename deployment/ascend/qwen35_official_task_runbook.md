# Qwen3.5-0.8B + MindSpeed-MM 单卡上机手册

更新时间：2026-07-22

## 1. 当前代码状态

项目已归档用户提供的 `MindSpeed-MM.zip`，校验信息见 `third_party/MindSpeed-MM.PROVENANCE.json`。压缩包包含官方 Qwen3.5 配置和一份 100 步参考日志，但不包含 Qwen3.5 权重、转换后的 DCP 权重或官方数据。

源压缩包中的 `mindspeed_mm/fsdp/models/qwen3_5/` 为空。项目已补充最小注册适配层：

```text
third_party/MindSpeed-MM/mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py
```

该适配层复用官方样例指定的 Transformers 5.2.0 `Qwen3_5ForConditionalGeneration`，只负责注册 `model_id=qwen3_5`，不伪造新的模型实现或优化结果。

## 2. 适用环境

- 1 张 Ascend 910 或 910B；
- 8-16 核 CPU、约 32 GB 内存；
- Ubuntu 22.04，仅在 ModelArts 所选官方镜像明确支持时使用；
- 优先使用官方预装镜像，不要在未知镜像上手工拼装 CANN；
- Qwen3.5 专项官方指南推荐 Python 3.10、PyTorch 2.7.1、`torch_npu` 2.7.1 和 CANN 8.5.2；实际 wheel 后缀及 ModelArts 镜像名称以赛题参考环境为准，禁止跨组合混装。

不要先按本地 macOS/CPU 依赖文件覆盖官方 NPU 镜像。先记录服务器现状，再创建隔离环境。

## 3. 上机前人工准备

1. 将当前项目上传到 ModelArts Notebook 持久化工作目录；
2. 下载 `Qwen/Qwen3.5-0.8B` Hugging Face 格式权重并记录来源、revision 和许可证；
3. 准备赛题提供的 `annotations_slim.json` 及其关联图片；仅有通用 COCO 压缩包不代表已经得到官方精度验证输入；
4. 确认有足够磁盘空间，并准备独立的模型、数据、日志目录；
5. 不把阿里云百炼 API Key 上传到 NPU 训练仓库。

可在服务器联网可用时下载官方 post-trained 权重：

```bash
python -m pip install -U huggingface_hub
hf download Qwen/Qwen3.5-0.8B --local-dir "$QWEN35_HF_DIR"
```

若 ModelArts 无法访问 Hugging Face，应在可信网络下载并校验后上传到 OBS/Notebook；不要改用 MLX、GGUF、ONNX 或第三方量化权重替代官方 Hugging Face safetensors。

## 4. 服务器首次检查

```bash
npu-smi info
python --version
python -c "import torch; print(torch.__version__)"
python -c "import torch_npu; print(torch_npu.__version__); print(torch.npu.is_available())"
python -c "import transformers; print(transformers.__version__)"
```

另行记录 CANN 版本、操作系统、ModelArts 镜像名称和实例规格。不要先执行本项目 `requirements.txt`，它是本机业务 Demo 依赖，不是 MindSpeed-MM NPU 环境文件。

## 5. 安装 MindSpeed-MM 配套依赖

参考包自身给出的 Qwen3.5 一键安装命令是：

```bash
cd "$MINDSPEED_MM_ROOT"
bash scripts/install.sh --msid eb10b92
bash examples/qwen3_5/install_extensions.sh
```

安装脚本会检查并可能替换 PyTorch/`torch_npu`。若 ModelArts 镜像已经提供配套环境，先记录版本，再根据脚本提示决定是否保留；不要无审查地确认覆盖。`install_extensions.sh` 最终要求 `transformers==5.2.0`、`triton-ascend==3.2.0`、`accelerate==1.2.0`。

安装后先执行代码静态预检：

```bash
cd "$PROJECT_ROOT"
python scripts/check_mindspeed_qwen35_bundle.py \
  --mindspeed-root "$MINDSPEED_MM_ROOT" \
  --static-only \
  --output results/ascend_qwen35/environment/server_preflight.json
```

该步骤应输出 `static_ready_runtime_unverified`。权重转换完成后，`scripts/run_mindspeed_qwen35.sh` 会自动执行包含 NPU、权重、DCP 和数据在内的严格动态预检；只有动态结果为 `ready_for_npu_run` 才进入训练。

## 6. 配置服务器路径

复制 `deployment/ascend/qwen35_server.env.example` 到服务器私有位置，修改所有绝对路径，然后载入：

```bash
set -a
source /home/ma-user/work/qwen35_server.env
set +a
cd "$PROJECT_ROOT"
```

环境文件不含 API Key。百炼在线 API 与本地 NPU 训练是两条不同链路。

## 7. 权重转换

首次运行需要把 Hugging Face 权重转换为 MindSpeed-MM DCP：

```bash
cd "$PROJECT_ROOT"
bash scripts/convert_qwen35_weights.sh
```

验收点是 `$QWEN35_DCP_DIR/release` 存在，转换日志位于 `results/ascend_qwen35/conversion/convert_console.log`。

## 8. 100 步真实运行

```bash
cd "$PROJECT_ROOT"
bash scripts/run_mindspeed_qwen35.sh
```

脚本依次完成严格预检、路径化配置生成、环境证据采集、单 NPU 100 步训练、参考日志精度比较和第 50 至 100 步性能统计。每次运行使用独立目录：

```text
results/ascend_qwen35/runs/<RUN_ID>/
├── preflight.json
├── environment.log
├── qwen3_5_0.8B_run.yaml
├── config_metadata.json
├── train.log
├── validation.json
└── validation.log
```

## 9. 迁移与优化顺序

1. 动态预检通过；
2. 权重转换通过；
3. 使用官方 `examples/qwen3_5/qwen3_5_0.8B_config.yaml` 口径，仅生成服务器路径副本；
4. 如需排错，可单独生成短步冒烟配置，但不得作为精度验收；
5. 运行完整 100 步；
6. 保留每一步 loss、grad norm、elapsed time 和完整配置；
7. 与官方精度日志比较，loss 与 grad norm 相对绝对误差均小于 2%；
8. 统计第 50 至 100 步耗时，作为性能基线；
9. 每次只启用一项优化，重复精度与性能检查；
10. 最后再接入图织天网的结构化证据研判接口。

## 10. 不能修改的验收口径

- 框架原有日志格式；
- loss 计算配置；
- global batch size；
- 官方规定的数据、步数和统计区间。

如确需修改，必须另建实验配置，不能与官方验收结果混在同一表格。

## 11. 当前仓库能直接复用的内容

- `skill/providers.py`：Provider 协议、证据约束和失败降级；
- `skill/engine.py`：结构化研判报告与证据边界；
- `model_adapter/schemas.py`：统一 `RiskAssessmentResult`；
- `backend/app/routers/demo.py`：报告服务入口；
- `scripts/benchmark.py`：业务 API/adapter 基准框架。
- `scripts/check_mindspeed_qwen35_bundle.py`：严格环境与资产预检；
- `scripts/prepare_mindspeed_qwen35_config.py`：生成服务器路径配置，不覆盖官方 YAML；
- `scripts/convert_qwen35_weights.sh`：HF 到 DCP 转换；
- `scripts/run_mindspeed_qwen35.sh`：单卡 100 步运行；
- `scripts/analyze_qwen35_training_log.py`：精度和性能日志验收。

压缩包自带的 `logs/qwen35_0.8B_20260608_110844.log` 只属于官方参考样例。只有租用实例中新产生的完整日志才能写成团队的昇腾迁移证据。
