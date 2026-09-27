"""Single-NPU HF-initialized training; never compare with the legacy baseline."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def prepare_config(original, model, dataset, output):
    config = copy.deepcopy(original)
    training = config['training']
    if config['model']['model_id'] != 'qwen3_5':
        raise ValueError('Expected qwen3_5 configuration')
    if (training.get('init_model_with_meta_device') is not False
            or training.get('load') != ''
            or training.get('load_rank0_and_broadcast') is not False):
        raise ValueError('Input must already be the verified HF initialization configuration')
    if training.get('train_iters') != 100:
        raise ValueError('Expected verified 100-step configuration')
    config['model']['model_name_or_path'] = str(model)
    params = config['data']['dataset_param']
    params['preprocess_parameters']['model_name_or_path'] = str(model)
    params['basic_parameters'].update(dataset=str(dataset), dataset_dir=str(dataset.parent),
                                      cache_dir=str(output / 'cache'))
    training.update(save=str(output / 'checkpoints'), save_interval=100)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'model', 'dataset', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--confirm-training', action='store_true', required=True)
    args = parser.parse_args()
    import yaml
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Output exists; refusing to overwrite')
    source = args.bundle.resolve() / 'third_party/MindSpeed-MM'
    trainer = source / 'mindspeed_mm/fsdp/train/trainer.py'
    utils = source / 'mindspeed_mm/fsdp/utils/utils.py'
    if not trainer.is_file() or 'require_backward_grad_sync = is_last_step' not in utils.read_text():
        raise ValueError('Missing trainer or required DDP gradient synchronization fix')
    config = prepare_config(yaml.safe_load(args.config.read_text()), args.model.resolve(),
                            args.dataset.resolve(), output)
    import torch
    import torch_npu  # Registers the NPU device backend.
    if not torch.npu.is_available():
        raise RuntimeError('NPU unavailable; CPU fallback is forbidden')
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / 'train100.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nnodes=1', '--nproc_per_node=1', str(trainer), str(config_path)]
    receipt = {'command': command, 'status': 'running',
               'source_config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
               'scope': 'training execution only; no accuracy or inference acceptance'}
    receipt_path = output / 'training_receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    try:
        with (output / 'train.log').open('w') as log:
            result = subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'NON_MEGATRON': 'true', 'ASCEND_RT_VISIBLE_DEVICES': '0',
                     'PYTORCH_NPU_ALLOC_CONF': 'expandable_segments:True'}, check=False)
        receipt.update(returncode=result.returncode,
                       status='command_succeeded' if result.returncode == 0 else 'failed')
        return result.returncode
    except BaseException as exc:
        receipt.update(status='failed', error=type(exc).__name__)
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
