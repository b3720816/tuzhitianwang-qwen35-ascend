from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MINDSPEED_ROOT = ROOT / "third_party/MindSpeed-MM"
DEFAULT_OUTPUT = ROOT / "results/ascend_qwen35/environment/preflight.json"
REQUIRED_TRANSFORMERS_VERSION = "5.2.0"
REQUIRED_ACCELERATE_VERSION = "1.2.0"
REQUIRED_TRITON_ASCEND_VERSION = "3.2.1"
REQUIRED_TORCH_PREFIX = "2.7.1"


def resolve_path(value: str | None, default: Path | None = None) -> Path | None:
    if value is None:
        return default.resolve() if default else None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str, *, required: bool = True) -> None:
    checks.append({"name": name, "passed": bool(passed), "required": required, "detail": detail})


def command_result(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def static_checks(mindspeed_root: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    config_path = mindspeed_root / "examples/qwen3_5/qwen3_5_0.8B_config.yaml"
    plugin_path = mindspeed_root / "mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py"
    trainer_path = mindspeed_root / "mindspeed_mm/fsdp/train/trainer.py"
    reference_log = mindspeed_root / "logs/qwen35_0.8B_20260608_110844.log"
    provenance = ROOT / "third_party/MindSpeed-MM.PROVENANCE.json"

    add_check(checks, "mindspeed_root", mindspeed_root.is_dir(), str(mindspeed_root))
    add_check(checks, "official_qwen35_config", config_path.is_file(), str(config_path))
    add_check(checks, "fsdp_trainer", trainer_path.is_file(), str(trainer_path))
    add_check(checks, "qwen35_registration_adapter", plugin_path.is_file(), str(plugin_path))
    add_check(checks, "archive_provenance", provenance.is_file(), str(provenance))
    add_check(
        checks,
        "official_reference_log",
        reference_log.is_file(),
        f"{reference_log} (reference only, not project NPU evidence)",
        required=False,
    )

    contract: dict[str, Any] = {}
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        training = config.get("training", {})
        contract = {
            "model_id": config.get("model", {}).get("model_id"),
            "train_iters": training.get("train_iters"),
            "micro_batch_size": training.get("micro_batch_size"),
            "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
            "seed": training.get("seed"),
            "plugin": training.get("plugin"),
        }
        add_check(checks, "config_model_id", contract["model_id"] == "qwen3_5", str(contract["model_id"]))
        add_check(checks, "config_train_iters", contract["train_iters"] == 100, str(contract["train_iters"]))
        add_check(
            checks,
            "config_single_npu_global_batch_size",
            contract["micro_batch_size"] == 1 and contract["gradient_accumulation_steps"] == 8,
            f"micro={contract['micro_batch_size']}, accumulation={contract['gradient_accumulation_steps']}, global=8",
        )
    if plugin_path.is_file():
        plugin_text = plugin_path.read_text(encoding="utf-8")
        add_check(
            checks,
            "qwen35_registry_key",
            '@model_register.register("qwen3_5")' in plugin_text,
            "model_id qwen3_5 must be registered",
        )
        add_check(
            checks,
            "qwen35_transformers_source",
            "transformers.models.qwen3_5.modeling_qwen3_5" in plugin_text,
            "adapter must use the Transformers 5.2.0 implementation",
        )
    return {"config_contract": contract, "paths": {"config": str(config_path), "plugin": str(plugin_path)}}


def runtime_checks(mindspeed_root: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    versions = {
        "torch": package_version("torch"),
        "torch_npu": package_version("torch-npu") or package_version("torch_npu"),
        "transformers": package_version("transformers"),
        "accelerate": package_version("accelerate"),
        "triton_ascend": package_version("triton-ascend") or package_version("triton_ascend"),
        "mindspeed": package_version("mindspeed"),
    }
    add_check(
        checks,
        "transformers_version",
        versions["transformers"] == REQUIRED_TRANSFORMERS_VERSION,
        f"expected {REQUIRED_TRANSFORMERS_VERSION}, found {versions['transformers']}",
    )
    add_check(checks, "torch_installed", bool(versions["torch"]), str(versions["torch"]))
    add_check(checks, "torch_npu_installed", bool(versions["torch_npu"]), str(versions["torch_npu"]))
    add_check(
        checks,
        "torch_version",
        bool(versions["torch"] and versions["torch"].startswith(REQUIRED_TORCH_PREFIX)),
        f"expected {REQUIRED_TORCH_PREFIX}.x, found {versions['torch']}",
    )
    add_check(
        checks,
        "torch_npu_version",
        bool(versions["torch_npu"] and versions["torch_npu"].startswith(REQUIRED_TORCH_PREFIX)),
        f"expected {REQUIRED_TORCH_PREFIX}.x, found {versions['torch_npu']}",
    )
    add_check(
        checks,
        "accelerate_version",
        versions["accelerate"] == REQUIRED_ACCELERATE_VERSION,
        f"expected {REQUIRED_ACCELERATE_VERSION}, found {versions['accelerate']}",
    )
    add_check(
        checks,
        "triton_ascend_version",
        versions["triton_ascend"] == REQUIRED_TRITON_ASCEND_VERSION,
        f"expected {REQUIRED_TRITON_ASCEND_VERSION}, found {versions['triton_ascend']}",
    )
    add_check(checks, "mindspeed_installed", bool(versions["mindspeed"]), str(versions["mindspeed"]))

    npu_available = False
    import_error = None
    try:
        torch = importlib.import_module("torch")
        importlib.import_module("torch_npu")
        npu_available = bool(torch.npu.is_available())
    except Exception as exc:
        import_error = f"{type(exc).__name__}: {exc}"
    add_check(checks, "npu_available", npu_available, import_error or "torch.npu.is_available() returned true")

    registration_error = None
    registered_class = None
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(mindspeed_root))
        register = importlib.import_module("mindspeed_mm.fsdp.utils.register")
        register.import_plugin(["mindspeed_mm/fsdp/models/qwen3_5"])
        registered_class = register.model_register.get("qwen3_5")
    except Exception as exc:
        registration_error = f"{type(exc).__name__}: {exc}"
    finally:
        sys.path[:] = original_path
    add_check(
        checks,
        "qwen35_runtime_registration",
        registered_class is not None,
        registration_error or f"{registered_class.__module__}.{registered_class.__name__}",
    )
    return {"versions": versions, "npu_available": npu_available, "registration_error": registration_error}


def external_assets_checks(
    checks: list[dict[str, Any]], hf_model_dir: Path | None, dcp_dir: Path | None, dataset: Path | None
) -> dict[str, Any]:
    assets = {
        "hf_model_dir": str(hf_model_dir) if hf_model_dir else None,
        "dcp_dir": str(dcp_dir) if dcp_dir else None,
        "dataset": str(dataset) if dataset else None,
    }
    add_check(checks, "hf_model_weights", bool(hf_model_dir and hf_model_dir.is_dir()), str(hf_model_dir), required=True)
    add_check(checks, "dcp_checkpoint", bool(dcp_dir and (dcp_dir / "release").exists()), str(dcp_dir), required=True)
    add_check(checks, "official_dataset", bool(dataset and dataset.is_file()), str(dataset), required=True)
    return assets


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight the Qwen3.5-0.8B MindSpeed-MM competition bundle")
    parser.add_argument("--mindspeed-root", default=str(DEFAULT_MINDSPEED_ROOT))
    parser.add_argument("--hf-model-dir")
    parser.add_argument("--dcp-dir")
    parser.add_argument("--dataset")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    mindspeed_root = resolve_path(args.mindspeed_root)
    assert mindspeed_root is not None
    checks: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "static_only" if args.static_only else "npu_runtime",
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "static": static_checks(mindspeed_root, checks),
    }
    if args.static_only:
        report["assets"] = {
            "status": "not_checked_in_static_mode",
            "note": "The archive does not contain Qwen3.5 weights, a converted DCP checkpoint, or the official dataset.",
        }
        report["runtime"] = {"status": "not_checked_in_static_mode"}
        report["npu_smi"] = {"status": "not_checked_in_static_mode"}
    else:
        report["assets"] = external_assets_checks(
            checks,
            resolve_path(args.hf_model_dir),
            resolve_path(args.dcp_dir),
            resolve_path(args.dataset),
        )
        report["runtime"] = runtime_checks(mindspeed_root, checks)
        report["npu_smi"] = command_result(["npu-smi", "info"])
        add_check(
            checks,
            "npu_smi",
            report["npu_smi"]["returncode"] == 0,
            report["npu_smi"].get("stderr") or "npu-smi info succeeded",
        )
        cann_env = os.getenv("CANN_ENV")
        add_check(
            checks,
            "cann_environment",
            bool(os.getenv("ASCEND_HOME_PATH") or (cann_env and Path(cann_env).is_file())),
            f"ASCEND_HOME_PATH={os.getenv('ASCEND_HOME_PATH')}; CANN_ENV={cann_env}",
        )

    report["checks"] = checks
    failed_required = [item for item in checks if item["required"] and not item["passed"]]
    if failed_required:
        report["status"] = "static_failed" if args.static_only else "not_ready"
    else:
        report["status"] = "static_ready_runtime_unverified" if args.static_only else "ready_for_npu_run"
    report["failed_required_checks"] = [item["name"] for item in failed_required]
    report["truthfulness_boundary"] = (
        "Static readiness does not prove an NPU migration. The project may claim completion only after a new 100-step "
        "log, environment evidence, and precision comparison are saved from the rented Ascend instance."
    )

    output = resolve_path(args.output)
    assert output is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failed_required else 1


if __name__ == "__main__":
    raise SystemExit(main())
