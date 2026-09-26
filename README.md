# 图织天网 Qwen3.5 迁移源码

本包包含赛事 MindSpeed-MM 参考源码、项目辅助脚本及云端备份恢复的实际适配文件。
MANIFEST.json 逐文件区分 cloud_backup 与 local_reference_or_support_file；云端备份没有包含全部框架文件，不能声称整个 ZIP 是云端逐字快照。

## 文件优先级
云端备份中的 Python、Shell、YAML 与 provenance 文件覆盖同路径本地文件；不包含 pyc、历史 bak、权重、数据、虚拟环境、密钥和 .env。差异保存在 CLOUD_LOCAL_DIFF.patch。
成功运行配置另存 verified_run/successful_run.yaml，保留当时绝对路径作为证据，不应在别的机器直接照抄。
第三方 LICENSE 与 Third-Party Open Source Software Notice.txt 随源码保留。参考日志是官方参考，不是本团队实测。

## 复现顺序与限制
先检查 deployment/ascend 的依赖说明，再用 scripts/setup_qwen35_runtime.py 与 audit_qwen35_runtime.py 做运行探针；HF 权重转换入口是 scripts/convert_qwen35_weights.sh，训练入口是 scripts/run_mindspeed_qwen35.sh。
使用独立 RUN_ID，设置 PYTHON_BIN、QWEN35_HF_DIR、QWEN35_DCP_DIR、QWEN35_DATASET 后运行训练入口。默认预处理进程 1、DataLoader workers 0；路径通过配置生成器写为绝对路径。覆盖并发参数需显式设置 QWEN35_NUM_WORKERS 或 QWEN35_PREPROCESSING_NUM_WORKERS。
2026-09-25 的训练后复现修正覆盖两个辅助脚本，MANIFEST 单独标注来源；旧文件在 verified_run/original_sources，差异在 REPRODUCTION_REPAIR.patch。这不是原始云端运行代码快照。配置生成回归测试与完整成功 YAML 比较通过，但未重新训练。
需要另外取得权重、初始 DCP、标注及图片、MindSpeed Core 与对应依赖。运行前核对 CANN 9.0.0 与 Triton-Ascend 3.2.1。
本次仅完成源码语法与打包完整性检查，没有在新的 Ascend 实例执行此 ZIP；训练证据属于此前成功运行。
业务 Demo 与业务 Skill 独立交付；本包不宣称在线业务推理已接入 NPU。
