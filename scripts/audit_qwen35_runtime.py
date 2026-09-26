"""Collect all Qwen3.5 environment failures without stopping at the first import."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
NPU_PACKAGES = {"torch-npu", "triton-ascend", "triton"}
# Generic model-zoo requirements unused by this snapshot's Qwen image/text path.
UNUSED_MM = {"beartype", "bs4", "diffusers", "ftfy", "imageio", "imageio-ffmpeg",
             "orjson", "pandarallel", "qwen-vl-utils", "timm"}
MM_OVERRIDES = {"transformers": "5.2.0", "accelerate": "1.2.0", "peft": "0.18.1"}
IMPORTS = {
    "trainer": "mindspeed_mm.fsdp.train.trainer",
    "model_hub": "mindspeed_mm.fsdp.models.modelhub",
    "collator": "mindspeed_mm.fsdp.data.data_utils.func_utils.collator",
    "media": "mindspeed_mm.fsdp.data.data_utils.func_utils.mm_plugin",
    "dataset_plugin": "mindspeed_mm.fsdp.data.datasets.huggingface.qwen2vl_dataset",
    "config": "mindspeed_mm.config.config_manager",
    "checkpoint": "mindspeed_mm.fsdp.checkpoint.dcp_checkpointer",
    "optimizer": "mindspeed_mm.fsdp.optimizer.optimizer",
    "converter": "checkpoint.convert_cli",
    "stateful_loader": "torchdata.stateful_dataloader",
    "peft": "peft",
}


def normalize(name):
    return name.lower().replace("_", "-").replace(".", "-")


def lock_requirements(path):
    from packaging.requirements import Requirement
    return [Requirement(line.strip()) for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def version_checks(lock, cpu=False):
    from packaging.requirements import Requirement
    rows = []
    for req in lock_requirements(lock) + [Requirement("mindspeed==0.12.1"), Requirement("mindspeed-mm==0.1")]:
        if cpu and normalize(req.name) in NPU_PACKAGES:
            continue
        try:
            version = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            version = None
        rows.append({"name": req.name, "expected": str(req.specifier), "found": version,
                     "passed": version is not None and req.specifier.contains(version)})
    return rows


def dependency_checks():
    from packaging.requirements import Requirement
    unexpected, exceptions, external = [], [], []
    prefix = Path(sys.prefix).resolve()
    for dist in metadata.distributions():
        name = normalize(dist.metadata.get("Name", ""))
        location = Path(dist.locate_file("")).resolve()
        if not location.is_relative_to(prefix):
            external.append({"name": name, "location": str(location)})
            continue
        for raw in dist.requires or []:
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            try:
                found = metadata.version(req.name)
            except metadata.PackageNotFoundError:
                found = None
            if found is not None and req.specifier.contains(found):
                continue
            item = {"owner": name, "requirement": raw, "found": found}
            key = normalize(req.name)
            allowed = name == "mindspeed-mm" and (
                (key in UNUSED_MM and found is None) or
                (key in MM_OVERRIDES and found == MM_OVERRIDES[key]))
            if (name == "mindspeed" and key == "numpy" and found == "1.26.4"
                    and str(req.specifier) == "<=1.26.0"
                    and metadata.version("mindspeed") == "0.12.1"):
                allowed = True
                item["reason"] = "CANN 9.0.0 / Triton-Ascend 3.2.1 pins numpy==1.26.4; Core 0.12.1 legacy metadata override."
            (exceptions if allowed else unexpected).append(item)
    return {"passed": not unexpected, "unexpected": unexpected,
            "qwen_profile_exceptions": exceptions, "external_cann_metadata": external}


def execute(command, cwd, env, timeout=120):
    start = time.monotonic()
    try:
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                                text=True, timeout=timeout)
        return {"passed": result.returncode == 0, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr,
                "seconds": round(time.monotonic() - start, 2)}
    except subprocess.TimeoutExpired as exc:
        return {"passed": False, "error": f"Timeout after {timeout}s",
                "stdout": str(exc.stdout or ""), "stderr": str(exc.stderr or "")}
    except OSError as exc:
        return {"passed": False, "error": str(exc)}


def probe(name, device):
    import torch
    if name in IMPORTS:
        import importlib
        importlib.import_module(IMPORTS[name])
    elif name == "registry":
        from mindspeed_mm.fsdp.utils.register import import_plugin, model_register, data_register
        import_plugin(["mindspeed_mm/fsdp/models/qwen3_5",
                       "mindspeed_mm/fsdp/data/datasets/huggingface"])
        assert model_register.get("qwen3_5") is not None
        assert data_register.get("huggingface") is not None
    elif name == "media_ops":
        import numpy as np
        import av
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        from torchvision.ops import nms
        assert to_tensor(Image.new("RGB", (16, 16))).shape == (3, 16, 16)
        assert torch.from_numpy(np.ones(4, dtype=np.float32)).sum().item() == 4
        assert nms(torch.tensor([[0., 0., 1., 1.]]), torch.ones(1), 0.5).tolist() == [0]
        assert av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24")
    elif name == "dataset_io":
        from datasets import load_dataset
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            path.write_text(json.dumps([{"messages": [{"role": "user", "content": "test"}],
                                        "images": []}]))
            dataset = load_dataset("json", data_files=str(path), split="train", cache_dir=tmp)
            assert len(dataset) == 1
    elif name == "peft_ops":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(torch.nn.Sequential(torch.nn.Linear(8, 8)),
                               LoraConfig(r=2, target_modules=["0"]))
        model(torch.randn(2, 8)).sum().backward()
        assert any(p.grad is not None for n, p in model.named_parameters() if "lora_" in n)
    elif name == "converter_roundtrip":
        from safetensors.torch import save_file
        import torch.distributed.checkpoint as dcp
        from checkpoint.fsdp.generic_dcp_converter import GenericDCPConverter
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf = root / "hf"
            hf.mkdir()
            weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)
            save_file({"weight": weight}, str(hf / "model.safetensors"))
            GenericDCPConverter().hf_to_dcp(hf_dir=str(hf), dcp_dir=str(root / "dcp"))
            state = {"model": {"weight": torch.zeros_like(weight)}}
            dcp.load(state, checkpoint_id=str(root / "dcp/release"))
            torch.testing.assert_close(state["model"]["weight"], weight)
    elif name == "npu_ops":
        import torch_npu  # noqa: F401
        assert torch.npu.is_available()
        assert torch.npu.device_count() >= 1
        x = torch.arange(4, dtype=torch.float32, device="npu:0")
        assert (x * x).cpu().tolist() == [0., 1., 4., 9.]
        layer = torch.nn.Linear(16, 8).to("npu:0")
        optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-4)
        loss = layer(torch.ones(2, 16, device="npu:0")).float().square().mean()
        loss.backward()
        optimizer.step()
        torch.npu.synchronize()
        assert torch.isfinite(loss).item()
    elif name == "triton_ops":
        global tl
        import torch_npu  # noqa: F401
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(x, y, out, size: tl.constexpr, block: tl.constexpr):
            offset = tl.program_id(0) * block + tl.arange(0, block)
            mask = offset < size
            tl.store(out + offset, tl.load(x + offset, mask=mask) +
                     tl.load(y + offset, mask=mask), mask=mask)

        x = torch.arange(1024, dtype=torch.float32, device="npu:0")
        output = torch.empty_like(x)
        add_kernel[(4,)](x, x, output, 1024, 256)
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), 2 * x.cpu())
    elif name == "qwen_tiny":
        from transformers import Qwen3_5Config
        from mindspeed_mm.fsdp.models.qwen3_5.modeling_qwen3_5 import MindSpeedQwen3_5ForConditionalGeneration
        config = Qwen3_5Config(
            text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                             head_dim=16, linear_num_key_heads=1, linear_num_value_heads=2,
                             linear_key_head_dim=16, linear_value_head_dim=16,
                             layer_types=["linear_attention", "full_attention"],
                             rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                                              "partial_rotary_factor": 1.0, "mrope_section": [3, 3, 2]}),
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                               out_hidden_size=32, num_position_embeddings=16),
            image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
        model = MindSpeedQwen3_5ForConditionalGeneration(config).to(device)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=device)
        loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss
        loss.backward()
        assert torch.isfinite(loss).item()
        assert any(p.grad is not None for p in model.parameters())
        print("synthetic_text_loss", loss.item())
        model.zero_grad(set_to_none=True)
        tokens = torch.tensor([[1, 62, 60, 63, 5, 6, 7, 8]], device=device)
        labels = tokens.clone()
        labels[:, 1:4] = -100
        loss = model(input_ids=tokens, labels=labels, use_cache=False,
                     pixel_values=torch.randn(4, 3 * 2 * 16 * 16, device=device),
                     image_grid_thw=torch.tensor([[1, 2, 2]], device=device)).loss
        loss.backward()
        assert torch.isfinite(loss).item()
        assert any(p.grad is not None for p in model.model.visual.parameters())
        print("synthetic_image_text_loss", loss.item())
    else:
        raise ValueError(name)
    print("PASS", name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT)
    parser.add_argument("--core", type=Path)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--cpu", action="store_true", help="Local validation only; never reports NPU ready")
    parser.add_argument("--probe", help=argparse.SUPPRESS)
    args = parser.parse_args()
    headers = Path(sys.prefix) / ".native-python310/usr/include"
    if (headers / "python3.10/Python.h").is_file():
        paths = [str(headers / "python3.10"), str(headers)]
        paths += [p for p in os.environ.get("CPATH", "").split(os.pathsep) if p]
        os.environ["CPATH"] = os.pathsep.join(dict.fromkeys(paths))
    if args.probe:
        probe(args.probe, "cpu" if args.cpu else "npu:0")
        return 0
    bundle = args.bundle.resolve()
    mm = bundle / "third_party/MindSpeed-MM"
    core = (args.core or bundle / "third_party/MindSpeed-Core").resolve()
    env = dict(os.environ, NON_MEGATRON="true", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", PYTHONNOUSERSITE="1")
    env["PYTHONPATH"] = os.pathsep.join([str(mm), str(core), env.get("PYTHONPATH", "")])
    output = args.output or bundle / "results/ascend_qwen35/environment/dependency_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"scope": "Qwen3.5 image/text FSDP training and GenericDCP conversion",
              "profile": "cann900-triton321-v2",
              "mode": "local_cpu_only" if args.cpu else "ascend_runtime",
              "host": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
              "versions": version_checks(ROOT / "deployment/ascend/qwen35-linux-aarch64-py310.lock", args.cpu),
              "dependencies": dependency_checks(), "probes": {},
              "not_verified": ["real model weights and official dataset", "100-step FSDP training",
                               "reference precision comparison", "full-size NPU memory use"]}
    report["source_sha256"] = {
        str(path.relative_to(mm)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [mm / "mindspeed_mm/fsdp/train/trainer.py",
                     mm / "mindspeed_mm/fsdp/data/data_utils/func_utils/collator.py",
                     mm / "mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py"]}
    if args.cpu:
        report["not_verified"].append("all NPU/CANN/Triton execution")
    report["core_commit"] = execute(["git", "rev-parse", "HEAD"], core, env)
    names = list(IMPORTS) + ["registry", "media_ops", "dataset_io", "peft_ops", "converter_roundtrip", "qwen_tiny"]
    if not args.cpu:
        names += ["npu_ops", "triton_ops"]
    for name in names:
        print("CHECK " + name, flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--probe", name]
        if args.cpu:
            command.append("--cpu")
        result = execute(command, mm, env, timeout=240 if args.cpu or name in {"triton_ops", "qwen_tiny"} else 120)
        report["probes"][name] = result
        print(("PASS " if result["passed"] else "FAIL ") + name, flush=True)
        if not result["passed"]:
            for key in ("error", "stdout", "stderr"):
                if result.get(key):
                    print(f"{key}:\n{result[key]}", flush=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    report["probes"]["converter_cli"] = execute(
        [sys.executable, "checkpoint/convert_cli.py", "GenericDCPConverter", "hf_to_dcp", "--help"], mm, env)
    failed = [r["name"] for r in report["versions"] if not r["passed"]]
    failed += [name for name, result in report["probes"].items() if not result["passed"]]
    if not report["dependencies"]["passed"]:
        failed.append("dependency_conflicts")
    report["failed_checks"] = failed
    report["status"] = "failed" if failed else ("cpu_checks_passed_npu_unverified" if args.cpu
                                                else "environment_smoke_passed_training_unverified")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(report["status"], "report:", output)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
