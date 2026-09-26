from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "third_party/MindSpeed-MM/examples/qwen3_5/qwen3_5_0.8B_config.yaml"
DEFAULT_OUTPUT = ROOT / "results/ascend_qwen35/generated/qwen3_5_0.8B_run.yaml"


def resolved_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def require_path(path: Path, kind: str) -> None:
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"Required directory not found: {path}")


def build_config(
    template: dict[str, Any],
    *,
    hf_model_dir: Path,
    dcp_dir: Path,
    dataset_file: Path,
    cache_dir: Path,
    train_iters: int,
    num_workers: int,
    save_dir: Path | None,
) -> dict[str, Any]:
    config = deepcopy(template)
    dataset = config["data"]["dataset_param"]
    dataset["preprocess_parameters"]["model_name_or_path"] = str(hf_model_dir)
    dataset["basic_parameters"]["dataset_dir"] = str(dataset_file.parent)
    dataset["basic_parameters"]["dataset"] = str(dataset_file.resolve())
    dataset["basic_parameters"]["cache_dir"] = str(cache_dir)
    config["data"]["dataloader_param"]["num_workers"] = num_workers
    config["model"]["model_name_or_path"] = str(hf_model_dir)
    config["training"]["load"] = str(dcp_dir)
    config["training"]["train_iters"] = train_iters
    if save_dir is not None:
        config["training"]["save"] = str(save_dir)
    return config


def config_contract(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    return {
        "model_id": config["model"].get("model_id"),
        "micro_batch_size": training.get("micro_batch_size"),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
        "global_batch_size_single_npu": (
            int(training.get("micro_batch_size", 0)) * int(training.get("gradient_accumulation_steps", 0))
        ),
        "train_iters": training.get("train_iters"),
        "seed": training.get("seed"),
        "learning_rate": training.get("lr"),
        "precision": config.get("parallel", {}).get("fsdp_plan", {}).get("param_dtype"),
        "template": config["data"]["dataset_param"]["basic_parameters"].get("template"),
        "cutoff_len": config["data"]["dataset_param"]["basic_parameters"].get("cutoff_len"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a path-safe Qwen3.5 MindSpeed-MM run config")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--hf-model-dir", required=True)
    parser.add_argument("--dcp-dir", required=True)
    parser.add_argument("--dataset", required=True, help="Path to annotations_slim.json")
    parser.add_argument("--cache-dir", default="results/ascend_qwen35/cache")
    parser.add_argument("--save-dir")
    parser.add_argument("--train-iters", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--metadata-output")
    parser.add_argument("--allow-short-smoke", action="store_true")
    args = parser.parse_args()

    if args.train_iters != 100 and not args.allow_short_smoke:
        parser.error("Official validation requires --train-iters 100; use --allow-short-smoke only for smoke runs")
    if args.train_iters < 1:
        parser.error("--train-iters must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")

    template_path = resolved_path(args.template)
    hf_model_dir = resolved_path(args.hf_model_dir)
    dcp_dir = resolved_path(args.dcp_dir)
    dataset_file = resolved_path(args.dataset)
    cache_dir = resolved_path(args.cache_dir)
    save_dir = resolved_path(args.save_dir) if args.save_dir else None
    output = resolved_path(args.output)
    metadata_output = (
        resolved_path(args.metadata_output)
        if args.metadata_output
        else output.with_suffix(output.suffix + ".metadata.json")
    )

    require_path(template_path, "file")
    require_path(hf_model_dir, "dir")
    require_path(dcp_dir, "dir")
    require_path(dataset_file, "file")
    if not (dcp_dir / "release").exists():
        raise FileNotFoundError(f"DCP checkpoint must contain a release entry: {dcp_dir / 'release'}")

    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    config = build_config(
        template,
        hf_model_dir=hf_model_dir,
        dcp_dir=dcp_dir,
        dataset_file=dataset_file,
        cache_dir=cache_dir,
        train_iters=args.train_iters,
        num_workers=args.num_workers,
        save_dir=save_dir,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "smoke_config" if args.train_iters != 100 else "official_100_step_config",
        "template": str(template_path),
        "output": str(output),
        "paths": {
            "hf_model_dir": str(hf_model_dir),
            "dcp_dir": str(dcp_dir),
            "dataset": str(dataset_file),
            "cache_dir": str(cache_dir),
            "save_dir": str(save_dir) if save_dir else None,
        },
        "contract": config_contract(config),
        "truthfulness_boundary": "A generated config is not evidence of an Ascend NPU run.",
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
