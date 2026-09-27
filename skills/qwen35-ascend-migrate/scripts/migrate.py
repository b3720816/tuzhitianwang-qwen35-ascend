"""Fail-closed orchestration of the verified Ascend runner; no package installation."""
import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'python', 'hf-model', 'dataset', 'cann-env', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--dcp', type=Path, help='Required only for legacy asset preflight')
    p.add_argument('--verified-config', type=Path, help='Verified HF-initialized training YAML')
    p.add_argument('--stage', choices=('preflight', 'train'), default='preflight')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--confirm-training', action='store_true')
    return p


def build_plan(args):
    paths = {key: getattr(args, key).expanduser().resolve() for key in
             ('bundle', 'python', 'hf_model', 'dataset', 'cann_env', 'output')}
    for key in ('bundle', 'hf_model'):
        if not paths[key].is_dir():
            raise ValueError('Missing directory: ' + str(paths[key]))
    for path in (paths['python'], paths['dataset'], paths['cann_env'],
                 paths['hf_model'] / 'config.json'):
        if not path.is_file():
            raise ValueError('Missing file: ' + str(path))
    if not os.access(paths['python'], os.X_OK):
        raise ValueError('Python is not executable')
    if args.execute and paths['output'].exists():
        raise ValueError('Output exists; choose a new directory')
    if args.execute and args.stage == 'train' and not args.confirm_training:
        raise ValueError('Training requires --confirm-training')
    env = {
        'PYTHON_BIN': str(paths['python']), 'CANN_ENV': str(paths['cann_env']),
        'QWEN35_HF_DIR': str(paths['hf_model']),
        'QWEN35_DATASET': str(paths['dataset']), 'RESULT_DIR': str(paths['output']),
        'MINDSPEED_MM_ROOT': str(paths['bundle'] / 'third_party/MindSpeed-MM'),
    }
    if args.stage == 'train':
        if args.verified_config is None or not args.verified_config.is_file():
            raise ValueError('Training requires --verified-config pointing to the verified train100.yaml')
        command = ['bash', '-c', 'source "$1" && shift && exec "$@"', 'hf-training',
                   str(paths['cann_env']), str(paths['python']),
                   str(Path(__file__).with_name('run_hf_training.py').resolve()),
                   '--bundle', str(paths['bundle']), '--model', str(paths['hf_model']),
                   '--dataset', str(paths['dataset']),
                   '--config', str(args.verified_config.resolve()),
                   '--output', str(paths['output'] / 'training'), '--confirm-training']
    else:
        if args.dcp is None or not (args.dcp / 'release').exists():
            raise ValueError('Legacy preflight requires --dcp with a release entry')
        env['QWEN35_DCP_DIR'] = str(args.dcp.resolve())
        if not (paths['bundle'] / 'scripts/check_mindspeed_qwen35_bundle.py').is_file():
            raise ValueError('Missing bundle preflight checker')
        command = ['bash', '-c', 'source "$1" && shift && exec "$@"', 'preflight',
                   str(paths['cann_env']), str(paths['python']),
                   str(paths['bundle'] / 'scripts/check_mindspeed_qwen35_bundle.py'),
                   '--mindspeed-root', env['MINDSPEED_MM_ROOT'],
                   '--hf-model-dir', env['QWEN35_HF_DIR'], '--dcp-dir', env['QWEN35_DCP_DIR'],
                   '--dataset', env['QWEN35_DATASET'],
                   '--output', str(paths['output'] / 'preflight.json')]
    return {'stage': args.stage, 'command': command, 'environment': env,
            'output': str(paths['output']), 'cwd': str(paths['bundle'])}


def execute(plan):
    out = Path(plan['output'])
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(plan, status='running', started_at=datetime.now(timezone.utc).isoformat())
    receipt_path = out / 'skill_receipt.json'
    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    save()
    try:
        with (out / 'skill_console.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(plan['command'], cwd=plan['cwd'],
                                    env={**os.environ, **plan['environment']},
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        receipt.update(returncode=result.returncode,
                       status='command_succeeded' if result.returncode == 0 else 'failed')
    except BaseException as exc:
        receipt.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                       error=type(exc).__name__)
        raise
    finally:
        receipt['finished_at'] = datetime.now(timezone.utc).isoformat()
        save()
    return result.returncode


def main():
    args = parser().parse_args()
    try:
        plan = build_plan(args)
        print(json.dumps(dict(plan, mode='execute' if args.execute else 'plan'), indent=2), flush=True)
        return execute(plan) if args.execute else 0
    except (ValueError, OSError) as exc:
        print(str(exc))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
