import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'skills/qwen35-ascend-migrate/scripts/run_hf_training.py'
spec = importlib.util.spec_from_file_location('hf_runner', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class HFRunnerTests(unittest.TestCase):
    def config(self):
        return {'model': {'model_id': 'qwen3_5'}, 'training': {
            'init_model_with_meta_device': False, 'load': '',
            'load_rank0_and_broadcast': False, 'train_iters': 100},
            'data': {'dataset_param': {'preprocess_parameters': {}, 'basic_parameters': {}}}}

    def test_paths_isolated_and_original_unchanged(self):
        original = self.config()
        new = runner.prepare_config(original, Path('/hf'), Path('/data/a.json'), Path('/new'))
        self.assertEqual(new['training']['save'], '/new/checkpoints')
        self.assertEqual(new['data']['dataset_param']['basic_parameters']['cache_dir'], '/new/cache')
        self.assertNotIn('save', original['training'])
        self.assertEqual(new['training']['load'], '')

    def test_legacy_initialization_rejected(self):
        for key, value in [('load', '/old/dcp'), ('load', None),
                           ('init_model_with_meta_device', True),
                           ('load_rank0_and_broadcast', True), ('train_iters', 5)]:
            config = self.config()
            config['training'][key] = value
            with self.assertRaises(ValueError):
                runner.prepare_config(config, Path('/hf'), Path('/data/a'), Path('/new'))
