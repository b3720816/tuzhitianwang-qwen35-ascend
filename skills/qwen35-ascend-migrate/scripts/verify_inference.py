"""Offline, single-NPU text smoke test, optionally restoring trusted DCP weights."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--mindspeed-root', type=Path)
    p.add_argument('--trust-local-checkpoint', action='store_true')
    p.add_argument('--prompt', default='Please briefly describe what a computer does.')
    p.add_argument('--max-new-tokens', type=int, default=32)
    p.add_argument('--execute', action='store_true')
    return p


def preflight(args):
    args.model = args.model.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not 1 <= args.max_new_tokens <= 128:
        raise ValueError('max-new-tokens must be between 1 and 128')
    if not args.prompt.strip() or len(args.prompt) > 4000:
        raise ValueError('Provide a nonempty prompt up to 4000 characters')
    config = json.loads((args.model / 'config.json').read_text())
    if config.get('model_type') != 'qwen3_5':
        raise ValueError('Expected model_type=qwen3_5')
    if not list(args.model.glob('*.safetensors')):
        raise ValueError('No local safetensors weights found')
    if args.output.exists():
        raise ValueError('Output exists; choose a new path')
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
        if not (args.checkpoint / '.metadata').is_file():
            raise ValueError('Pass the exact DCP iteration directory containing .metadata')
        if not list(args.checkpoint.glob('*.distcp')):
            raise ValueError('No DCP shards found')
        if args.mindspeed_root is None:
            raise ValueError('DCP restore requires --mindspeed-root')
        args.mindspeed_root = args.mindspeed_root.expanduser().resolve()
        if not (args.mindspeed_root / 'checkpoint/common/merge_dcp_to_hf.py').is_file():
            raise ValueError('Missing bundled DCP reader')
        if args.execute and not args.trust_local_checkpoint:
            raise ValueError('DCP metadata uses pickle; require --trust-local-checkpoint for your own backup')
    return {'mode': 'dcp_restore' if args.checkpoint else 'base_model',
            'model': str(args.model),
            'checkpoint': str(args.checkpoint) if args.checkpoint else None,
            'output': str(args.output), 'device': 'npu:0',
            'execution_requested': args.execute,
            'scope': 'text generation only; not multimodal accuracy or optimizer resume'}


def run(args, report):
    # A failed import or runtime probe still leaves a receipt, never a success marker.
    args.output.mkdir(parents=True, exist_ok=False)
    report.update(status='running', started_at=datetime.now(timezone.utc).isoformat(),
                  python=sys.version, executable=sys.executable, host=platform.node(),
                  prompt=args.prompt, max_new_tokens=args.max_new_tokens)
    result_path = args.output / 'inference_report.json'
    def save():
        result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=sorted) + '\n')
    save()
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        import torch_npu
        import transformers
        from transformers import AutoTokenizer
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
        report['versions'] = {'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
                              'transformers': transformers.__version__}
        if not torch.npu.is_available():
            raise RuntimeError('NPU unavailable; CPU fallback is prohibited')
        torch.npu.set_device(0)
        torch.manual_seed(42)
        files = [args.model / 'config.json', *sorted(args.model.glob('*.safetensors'))]
        if args.checkpoint:
            files += [args.checkpoint / '.metadata', *sorted(args.checkpoint.glob('*.distcp'))]
        report['source_files'] = [{'path': str(p), 'bytes': p.stat().st_size,
                                   'sha256': sha256(p)} for p in files]
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True,
                                                  trust_remote_code=False)
        model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(args.model), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation='eager', output_loading_info=True)
        report['loading_info'] = loading
        if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise RuntimeError('Base model loading was not clean; inspect loading_info')
        if args.checkpoint:
            sys.path.insert(0, str(args.mindspeed_root))
            from checkpoint.common.merge_dcp_to_hf import load_dcp_state_dict
            state = load_dcp_state_dict(str(args.checkpoint))
            model.load_state_dict(state, strict=True)
            del state
            report['checkpoint_weights_strictly_loaded'] = True
        model = model.to('npu:0').eval()
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': args.prompt}],
                                              tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors='pt').to('npu:0')
        if inputs['input_ids'].shape[1] > 1024:
            raise ValueError('Prompt exceeds 1024 tokens')
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[:, -1, :]
            if not torch.isfinite(logits).all().item():
                raise RuntimeError('Non-finite logits')
            del logits
            torch.npu.synchronize()
            start = time.monotonic()
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            torch.npu.synchronize()
            report['generation_seconds'] = time.monotonic() - start
        generated = output[0, inputs['input_ids'].shape[1]:].cpu().tolist()
        answer = tokenizer.decode(generated, skip_special_tokens=True)
        report.update(generated_token_ids=generated, answer=answer, finite_logits=True)
        if not generated or not answer.strip():
            raise RuntimeError('No nonempty generated text')
        report['status'] = 'passed'
        return 0
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      error=f'{type(exc).__name__}: {exc}')
        (args.output / 'error.log').write_text(traceback.format_exc())
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        save()


def main():
    args = get_parser().parse_args()
    try:
        report = preflight(args)
    except (OSError, ValueError) as exc:
        print(f'Preflight failed: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2), flush=True)
    return run(args, report) if args.execute else 0


if __name__ == '__main__':
    raise SystemExit(main())
