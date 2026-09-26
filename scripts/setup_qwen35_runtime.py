"""Install the audited Qwen3.5 profile without changing the existing environment."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ASCEND_INDEX = "https://triton-ascend.osinfra.cn/pypi/simple"


def pip_environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith("PIP_")}
    env.update(PYTHONPATH="", PYTHONNOUSERSITE="1", PIP_CONFIG_FILE=os.devnull,
               PIP_DISABLE_PIP_VERSION_CHECK="1")
    env.pop("PYTHONHOME", None)
    return env


def validate_host(bundle, venv, cann):
    if platform.system() != "Linux" or platform.machine() != "aarch64" or sys.version_info[:2] != (3, 10):
        raise RuntimeError("Run with the HiDevLab Linux aarch64 Python 3.10 interpreter, not the local computer.")
    if not cann.is_file() or "cann-9.0.0" not in str(cann.resolve()):
        raise RuntimeError("This profile requires the verified /usr/local/Ascend/cann-9.0.0 installation.")
    if venv.resolve() in {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}:
        raise RuntimeError("Refusing to change the bootstrap/system environment.")
    marker = venv / ".qwen35-managed.json"
    if venv.exists() and not marker.is_file():
        raise RuntimeError(f"Refusing to modify an unrelated existing directory: {venv}")
    if marker.is_file() and json.loads(marker.read_text()).get("owner") != "qwen35-runtime-setup":
        raise RuntimeError("Target environment ownership marker does not match.")
    for relative in ["third_party/MindSpeed-Core/mindspeed/fsdp/utils/log.py",
                     "third_party/MindSpeed-MM/mindspeed_mm/fsdp/train/trainer.py",
                     "third_party/MindSpeed-MM/mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py"]:
        if not (bundle / relative).is_file():
            raise FileNotFoundError(bundle / relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT)
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann/set_env.sh"))
    parser.add_argument("--index-url", default="https://repo.huaweicloud.com/repository/pypi/simple")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    venv = (args.venv or bundle.parent / ".venv-qwen35-checked").absolute()
    validate_host(bundle, venv, args.cann)
    if not args.index_url.startswith("https://"):
        parser.error("Use an HTTPS package index.")
    lock = ROOT / "deployment/ascend/qwen35-linux-aarch64-py310.lock"
    env = pip_environment()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    evidence = bundle / "results/ascend_qwen35/environment" / f"setup-{stamp}"
    evidence.mkdir(parents=True, exist_ok=False)
    wheelhouse = evidence / "wheels"
    log_path = evidence / "setup.log"
    state = {"status": "running", "target": str(venv), "source_bundle": str(bundle),
             "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()}
    state_path = evidence / "setup.json"

    def run(stage, command, cwd=None):
        state["stage"] = stage
        state_path.write_text(json.dumps(state, indent=2))
        print(f"\n[{stage}]", flush=True)
        with log_path.open("a") as log:
            log.write(f"\n[{stage}]\n")
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        if code:
            raise RuntimeError(f"{stage} failed (exit {code}); complete log: {log_path}")

    try:
        pip = [sys.executable, "-m", "pip"]
        run("download-all-before-install", pip + ["download", "--only-binary=:all:",
            "--index-url", args.index_url, "--extra-index-url", ASCEND_INDEX,
            "--dest", str(wheelhouse), "-r", str(lock)])
        if not venv.exists():
            venv.mkdir(parents=True)
            (venv / ".qwen35-managed.json").write_text(json.dumps({"owner": "qwen35-runtime-setup"}))
        run("create-isolated-environment", [sys.executable, "-m", "venv", "--without-pip", str(venv)])
        python = venv / "bin/python"
        target_pip = pip + ["--python", str(python)]
        offline = ["--no-index", "--find-links", str(wheelhouse), "-r", str(lock)]
        run("native-resolver-check", target_pip + ["install", "--dry-run", "--report",
            str(evidence / "install-plan.json")] + offline)
        run("install-locked-dependencies", target_pip + ["install"] + offline)
        # Both distributions own triton/ files: Ascend must be installed last.
        for package in ["triton==3.5.0", "triton-ascend==3.2.1"]:
            run("ordered-install-" + package, target_pip + ["install", "--no-index",
                "--find-links", str(wheelhouse), "--no-deps", "--force-reinstall", package])
        for project in ["MindSpeed-Core", "MindSpeed-MM"]:
            source = bundle / "third_party" / project
            run(f"install-{project}", target_pip + ["install", "--no-index", "--no-deps",
                "--no-build-isolation", "-e", str(source)], cwd=source)
        env["TRITON_CACHE_DIR"] = str(evidence / "triton-cache")
        run("all-runtime-checks", ["bash", "-c", 'source "$1" || exit $?; shift; exec "$@"', "qwen35-audit",
            str(args.cann), str(python), str(ROOT / "scripts/audit_qwen35_runtime.py"),
            "--bundle", str(bundle), "--output", str(evidence / "dependency_audit.json")])
        state["status"] = "environment_smoke_passed_training_unverified"
        print(f"\nChecked Python: {python}\nReport: {evidence / 'dependency_audit.json'}")
        print("Model assets, real-data training and precision comparison still require validation.")
        return 0
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
        print(str(exc), file=sys.stderr)
        print(f"Report directory: {evidence}\nYour original environment was not modified.", file=sys.stderr)
        return 1
    finally:
        state_path.write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
