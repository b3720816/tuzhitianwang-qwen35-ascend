"""Fixed 20-image functional comparison; answers require human review."""
import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    b = root / 'qwen35_ascend_server_bundle'
    model_dir = root / 'models/Qwen3.5-0.8B'
    data_dir = root / 'data/official_coco/extracted/dataset'
    annotation = data_dir / 'annotations_slim.json'
    checkpoint = b / 'results/ascend_qwen35/runs/hf_init_full_20260927_081557/checkpoints/iter_0000100'
    report = dict(status='running', scope='20 fixed candidate holdout images; manual review required; not official accuracy',
                  annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
                  checkpoint=str(checkpoint), model=str(model_dir), device='npu:0',
                  max_new_tokens=96, do_sample=False, rows=[])

    def save():
        temp = out / 'comparison.json.tmp'
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        temp.replace(out / 'comparison.json')

    save()
    try:
        import torch
        import torch_npu
        from PIL import Image
        from transformers import AutoProcessor
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
        assert torch.npu.is_available(), 'NPU unavailable'
        torch.npu.set_device(0)
        torch.manual_seed(42)
        data = json.loads(annotation.read_text())
        train_images = {str((data_dir / name).resolve()) for row in data[:1000] for name in row['images']}
        for index in range(1000, 1020):
            row = data[index]
            assert len(row['images']) == 1, 'Expected single image'
            image = (data_dir / row['images'][0]).resolve()
            assert str(image) not in train_images and image.is_file()
            messages = row['messages']
            assert messages[0]['role'] == 'user' and messages[1]['role'] == 'assistant'
            report['rows'].append(dict(index=index, image=str(image),
                image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                question=messages[0]['content'].replace('<image>', '').strip(),
                reference_answer=messages[1]['content'], manual_base_score=None,
                manual_trained_score=None))
        save()
        processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(model_dir), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation='eager').to('npu:0').eval()
        sys.path.insert(0, str(b / 'third_party/MindSpeed-MM'))
        from checkpoint.common.merge_dcp_to_hf import load_dcp_state_dict
        for mode in ('base', 'trained'):
            if mode == 'trained':
                state = load_dcp_state_dict(str(checkpoint))
                model.load_state_dict(state, strict=True)
                del state
                report['checkpoint_weights_strictly_loaded'] = True
            for row in report['rows']:
                messages = [{'role': 'user', 'content': [
                    {'type': 'image'}, {'type': 'text', 'text': row['question']}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                with Image.open(row['image']) as image:
                    inputs = processor(text=[text], images=[image.convert('RGB')],
                                       return_tensors='pt').to('npu:0')
                assert 'pixel_values' in inputs
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=96, do_sample=False)
                tokens = output[0, inputs['input_ids'].shape[1]:].cpu().tolist()
                answer = processor.batch_decode([tokens], skip_special_tokens=True)[0]
                row[mode + '_answer'] = answer
                row[mode + '_tokens'] = tokens
                row[mode + '_nonempty'] = bool(answer.strip())
                row[mode + '_length_limit_reached'] = len(tokens) >= 96
                save()
                print(f"[{mode} {row['index']}] {answer!r}", flush=True)
                del inputs, output
        report['status'] = 'completed_pending_manual_review'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        (out / 'error.log').write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        save()
    print('REPORT:', out / 'comparison.json', flush=True)
    return 0 if report['status'] == 'completed_pending_manual_review' else 1


if __name__ == '__main__':
    raise SystemExit(main())
