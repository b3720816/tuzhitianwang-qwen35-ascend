"""Audit HF initialization through the real trainer before optimizer creation."""
import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path


def unwrap_for_audit(model, ddp_type):
    return model.module if isinstance(model, ddp_type) else model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mindspeed-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('This audit supports one NPU only')
    root, config, model_dir, out = (p.resolve() for p in
                                    (args.mindspeed_root, args.config, args.model, args.output))
    if not config.is_file() or not (model_dir / 'config.json').is_file():
        parser.error('Missing source configuration or model config')
    out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'training_steps': 0, 'optimizer_created': False,
              'source_config': str(config), 'source_model': str(model_dir),
              'source_config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
              'runtime_overrides': {'init_model_with_meta_device': False, 'load': ''},
              'scope': 'parameter initialization only, not training or inference acceptance'}
    def save():
        (out / 'initialization_report.json').write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    save()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(root))
    sys.argv = [str(root / 'mindspeed_mm/fsdp/train/trainer.py'), str(config)]
    try:
        import torch
        import torch_npu
        from safetensors import safe_open
        from mindspeed_mm.fsdp.train.trainer import Trainer, Arguments, ConfigManager

        class AuditComplete(Exception):
            pass

        class AuditTrainer(Trainer):
            def get_optimizer(self):
                # Trainer calls this directly after the original get_model pipeline.
                source = {}
                for file in sorted(model_dir.glob('*.safetensors')):
                    with safe_open(str(file), framework='pt', device='cpu') as handle:
                        for key in handle.keys():
                            if key in source:
                                raise RuntimeError('Duplicate HF key: ' + key)
                            source[key] = file
                cfg = json.loads((model_dir / 'config.json').read_text())
                if cfg.get('text_config', {}).get('tie_word_embeddings') is not True:
                    raise RuntimeError('Expected tied text embeddings')
                emb = 'model.language_model.embed_tokens.weight'
                audited_model = unwrap_for_audit(self.model, torch.nn.parallel.DistributedDataParallel)
                report['parallel_wrapper'] = type(self.model).__name__
                inp = audited_model.get_input_embeddings().weight
                head = audited_model.get_output_embeddings().weight
                report['shared_parameter_identity'] = inp is head
                matched, mismatches, absent, conversions = [], [], [], []
                for key, tensor in audited_model.state_dict().items():
                    source_key = emb if key == 'lm_head.weight' and key not in source else key
                    if source_key not in source:
                        absent.append(key)
                        continue
                    with safe_open(str(source[source_key]), framework='pt', device='cpu') as handle:
                        expected = handle.get_tensor(source_key)
                    actual = tensor.full_tensor() if hasattr(tensor, 'full_tensor') else tensor
                    actual = actual.detach().cpu()
                    if actual.dtype != expected.dtype:
                        conversions.append({'key': key, 'source_dtype': str(expected.dtype),
                                            'actual_dtype': str(actual.dtype)})
                    if actual.shape != expected.shape:
                        mismatches.append({'key': key, 'reason': 'shape'})
                    elif not torch.equal(actual, expected.to(dtype=actual.dtype)):
                        mismatches.append({'key': key, 'reason': 'values',
                                           'max_abs_diff': (actual.float() - expected.to(dtype=actual.dtype).float()).abs().max().item()})
                    else:
                        matched.append(key)
                    del actual, expected
                report.update(matched_count=len(matched), mismatches=mismatches,
                              comparison='exact equality after source cast to actual dtype; no tolerance',
                              dtype_conversions=conversions,
                              missing_source_keys=absent,
                              unused_source_keys=sorted(set(source) - set(audited_model.state_dict())))
                report['status'] = ('passed' if matched and not mismatches and not absent
                                    and report['shared_parameter_identity'] else 'failed')
                raise AuditComplete()

        parsed = ConfigManager(config_class=Arguments).load_and_parse()
        parsed.training.init_model_with_meta_device = False
        parsed.training.load = ''
        parsed.training.load_rank0_and_broadcast = False
        parsed.training.save = str(out / 'unused_checkpoints')
        parsed.model.model_name_or_path = str(model_dir)
        if parsed.training.lora.enable:
            raise RuntimeError('This audit does not support LoRA')
        try:
            AuditTrainer(parsed)
        except AuditComplete:
            pass
        else:
            raise RuntimeError('Audit failed to stop before optimizer creation')
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        (out / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        save()
        torch_module = sys.modules.get('torch')
        if torch_module and torch_module.distributed.is_initialized():
            torch_module.distributed.destroy_process_group()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
