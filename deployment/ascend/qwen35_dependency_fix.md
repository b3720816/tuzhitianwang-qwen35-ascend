# HiDevLab Qwen3.5 依赖修复包

## 2026-09-23 CANN 9.0.0 修订

服务器已实际通过除 Triton 外的运行检查。首次 Triton 编译缺少 Python.h，补齐头文件后暴露 `RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE` 编译错误。原因是旧归档固定的 Triton-Ascend 3.2.0 对应 CANN 8.5.0，不适合此实例的 CANN 9.0.0。之前保留旧归档版本的决定有误。

本修订使用官方 CANN 9.0.0 配套 Triton-Ascend 3.2.1。ARM64 wheel 从官方额外源 `https://triton-ascend.osinfra.cn/pypi/simple` 获取，实际文件由华为镜像提供。它依赖社区 Triton 3.5.0，安装器按社区包在先、昇腾包在后的顺序重装，避免共享文件被覆盖。

已核对 wheel 元数据要求：attrs 24.2.0、NumPy 1.26.4、SciPy 1.13.1、decorator 5.1.1、psutil 6.0.0、pytest 8.3.2、pytest-xdist 3.6.1。对应锁文件同步修订。MindSpeed-Core 0.12.1 的旧元数据 `numpy<=1.26.0` 与之不可同时满足，检查器只对该框架版本、该约束和实际 NumPy 1.26.4 记录显式例外；不豁免其他 NumPy 版本或要求。

已解压的 Python 头文件继续保存在 `.venv-qwen35-checked/.native-python310/usr/include`，检查器自动加入编译器搜索路径。没有头文件时可运行 `bash scripts/repair_qwen35_python_headers.sh`，它只下载并解压 Ubuntu 开发包，不安装或升级系统包。后续正式训练仍需导出这两个 CPATH 目录。

旧失败报告保留不变。本修订必须重新产生 NPU 检查报告，不能据本机 CPU 测试宣布服务器通过。

修订后复核：99 个 Linux aarch64 / Python 3.10 二进制包联合解析通过；本机隔离环境 98 项版本检查、18 项功能检查、17 项测试通过。报告为 `build/qwen35_dependency_audit/cpu310-cann900-v2-audit.json`，状态仍明确为 `cpu_checks_passed_npu_unverified`。新包为 `qwen35_cann900_fix.zip`，不要再解压旧修复包覆盖新锁文件。

## 这次修复什么

之前的安装遗漏了训练入口的间接依赖。`pip install --no-deps` 只安装框架本身，不会补齐 Pydantic、TorchData、Pillow、PEFT、PyAV、Datasets 等；仅验证模型注册成功，也不能说明训练入口可运行。

同时，压缩包中 MindSpeed-MM 的通用依赖表固定 Transformers 4.57.0、Accelerate 0.32.1 和 PEFT 0.7.1，而 Qwen3.5 专用脚本使用 Transformers 5.2.0、Accelerate 1.2.0。不能原样混装，也不能看到红字就一律降回通用版本。

此补丁针对本包的 **Qwen3.5 图像/文本 FSDP 路径及 GenericDCP 权重转换**，不是整个 MindSpeed-MM 模型库所有语音、视频生成模型的通用环境。

## 执行

将 `qwen35_dependency_fix.zip` 上传到服务器 `/workspace/tuzhitianwang/`，然后在 HiDevLab 终端依次执行以下两条完整命令：

```bash
unzip -o /workspace/tuzhitianwang/qwen35_dependency_fix.zip -d /workspace/tuzhitianwang/
/workspace/tuzhitianwang/.venv/bin/python /workspace/tuzhitianwang/qwen35_ascend_server_bundle/scripts/setup_qwen35_runtime.py
```

程序会先下载并核对全部固定版本，再安装到 `/workspace/tuzhitianwang/.venv-qwen35-checked`。原 `.venv`、系统 Python、驱动和 CANN 不改动，不运行 apt，不删除任何已有环境，不启动正式训练。已存在但不属于本安装器的目标目录会被拒绝使用。

安装结束后，只有状态为 `environment_smoke_passed_training_unverified` 才表示本轮环境检查通过。所有检查独立运行，单项失败不会阻止其他检查，完整输出在：

```text
/workspace/tuzhitianwang/qwen35_ascend_server_bundle/results/ascend_qwen35/environment/setup-时间戳/
```

其中 `setup.log` 是安装日志，`install-plan.json` 是原生服务器解析结果，`dependency_audit.json` 汇总所有检查及完整异常。后续排查以这一份汇总报告为依据，不需要逐库运行命令、逐个截屏。

通过后切换到新环境，后续不要再使用旧 `.venv/bin/python`：

```bash
source /workspace/tuzhitianwang/.venv-qwen35-checked/bin/activate
source /usr/local/Ascend/cann/set_env.sh
export NON_MEGATRON=true
export PYTHON_BIN=/workspace/tuzhitianwang/.venv-qwen35-checked/bin/python
```

## 固定版本

全量 99 个直接及传递依赖见 `qwen35-linux-aarch64-py310.lock`，全部精确固定版本。主要组合：

| 组件 | 版本 |
| --- | --- |
| 操作系统/解释器 | Linux aarch64、Python 3.10、CANN 9.0.0 |
| torch / torch-npu | 2.7.1 / 2.7.1.post4 |
| torchvision / torchdata | 0.22.1 / 0.11.0 |
| Transformers / Accelerate / PEFT | 5.2.0 / 1.2.0 / 0.18.1 |
| NumPy / Pandas / Datasets | 1.26.4 / 2.0.3 / 3.6.0 |
| Triton-Ascend / 社区 Triton (ARM64) | 3.2.1 / 3.5.0，与 CANN 9.0.0 配套 |
| MindSpeed-Core / MindSpeed-MM | 0.12.1 / 0.1，本地源码可编辑安装 |

MindSpeed-Core 核对源码为 `26.0.0_core_r0.12.1` 分支的 `dc6bfa96e4305c14b192aa004b5df8b94c2844d8`。安装器使用服务器已有源码，不拉取或覆盖分支。

## 核对范围与边界

2026-09-23 交付前验证：macOS ARM / Python 3.10.21 的隔离环境中，96 项版本核对（94 个通用依赖加两个源码框架）、18 项功能检查、15 项测试全部通过。服务器 Python 为 3.10.12，操作系统与 NPU 不同，因此服务器检查仍为待执行。首次冷导入曾触发 120 秒超时，保留了失败记录；增加本机检查超时后完整复测通过。

- 静态追踪训练入口、模型/数据插件、转换入口的顶层导入链，首轮覆盖 126 个源码模块。
- 按 Linux aarch64、glibc 2.35、Python 3.10 解析全量依赖；官方 PyPI 和用户实际使用的华为镜像均完成二进制包解析检查。
- 检查每一个锁定版本及传递依赖；分别导入训练器、模型工厂、collator、媒体模块、数据集插件、配置、checkpoint、优化器和转换器。
- 功能检查包括图像变换、视频帧构造、JSON 数据读取、PEFT 反向传播、微型 HF→DCP 转换后读取比对、微型 Qwen3.5 纯文本及图文前向/反向。
- 服务器额外执行 NPU 张量/优化步骤、Triton 实际编译与计算；不是只查看 `is_available()`。
- **不等于已完成真实 0.8B 权重训练、100 步日志、精度对比或性能验收。** 本地 CPU 测试不能替代 NPU、HCCL、FSDP、驱动/编译器验证。

通用 MindSpeed-MM 元数据仍声明其他模型需要的 `beartype、bs4、diffusers、ftfy、imageio、imageio-ffmpeg、orjson、pandarallel、qwen-vl-utils、timm`。当前图像/文本路径不加载这些包；通用表与专用版本的三处冲突也单独记录为 `qwen_profile_exceptions`。检查器只认可明确列出的例外，其他缺库/版本冲突全部判失败，**不会宣称通用 `pip check` 零冲突**。

微型模型可能提示未启用 FLA/causal-conv1d 快速路径。这是当前最小模型适配器使用 PyTorch 回退实现的限制，不应直接安装面向 CUDA 的包来消除提示。是否达到官方性能必须在真正 NPU 训练后判断。

Triton 版本以当前实例 CANN 的官方配套为准，不再沿用旧归档的 3.2.0。实际 NPU 编译和运行仍为必需检查；失败即阻止环境通过。

## 核对依据

- 本包 `third_party/MindSpeed-MM/pyproject.toml`、`examples/qwen3_5/install_extensions.sh` 和实际源码导入链。
- [TorchNPU 官方版本配套](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)：CANN 9.0.0 对应 Torch 2.7.1 + torch-npu 2.7.1.post4。
- [MindSpeed-Core 对应分支依赖](https://github.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/requirements.txt)。
- [PEFT 0.18.1 发布说明](https://github.com/huggingface/peft/releases/tag/v0.18.1)：包含 Transformers v5 兼容修复。
- [Triton-Ascend 官方安装说明](https://github.com/Ascend/triton-ascend/blob/main/docs/en/installation_guide.md)：CANN 9.0.0 对应 3.2.1，ARM64 依赖社区 Triton 3.5.0，并要求昇腾包覆盖安装在后。

依赖解析、CPU 运行和服务器 NPU 运行是不同级别的证据，报告中分开标记。
