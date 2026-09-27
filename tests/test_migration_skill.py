import importlib.util
import json
import os
import sys
import tempfile
import unittest
import venv
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'skills/qwen35-ascend-migrate/scripts/migrate.py'
spec = importlib.util.spec_from_file_location('migrate', SCRIPT)
migrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migrate)


class MigrationSkillTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'space and $literal'
        for name in ('bundle/scripts', 'hf', 'dcp/release'):
            (self.root / name).mkdir(parents=True)
        for name in ('bundle/scripts/check_mindspeed_qwen35_bundle.py',
                     'bundle/scripts/run_mindspeed_qwen35.sh', 'hf/config.json',
                     'annotations.json', 'cann.sh'):
            (self.root / name).touch()
        self.args = migrate.parser().parse_args([
            '--bundle', str(self.root / 'bundle'), '--python', sys.executable,
            '--hf-model', str(self.root / 'hf'), '--dcp', str(self.root / 'dcp'),
            '--dataset', str(self.root / 'annotations.json'),
            '--cann-env', str(self.root / 'cann.sh'), '--output', str(self.root / 'out')])

    def test_plan_does_not_write(self):
        plan = migrate.build_plan(self.args)
        self.assertIn(str((self.root / 'annotations.json').resolve()), plan['command'])
        self.assertFalse(self.args.output.exists())

    def test_missing_asset_rejected(self):
        (self.root / 'hf/config.json').unlink()
        with self.assertRaises(ValueError):
            migrate.build_plan(self.args)

    def test_symlink_interpreter_preserved_for_both_stages(self):
        python = self.root / 'venv/bin/python'
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        self.args.python = Path(os.path.relpath(python))
        self.args.verified_config = self.root / 'train100.yaml'
        self.args.verified_config.touch()
        for stage in ('preflight', 'train'):
            with self.subTest(stage=stage):
                self.args.stage = stage
                plan = migrate.build_plan(self.args)
                self.assertEqual(plan['environment']['PYTHON_BIN'], str(python))
                self.assertEqual(plan['command'][5], str(python))

    def test_preflight_runs_in_selected_virtual_environment(self):
        directory = self.root / 'selected-venv'
        venv.EnvBuilder(with_pip=False, symlinks=True).create(directory)
        self.args.python = directory / 'bin/python'
        checker = self.root / 'bundle/scripts/check_mindspeed_qwen35_bundle.py'
        checker.write_text('import sys\nprint(sys.prefix)\n')
        plan = migrate.build_plan(self.args)
        self.assertEqual(migrate.execute(plan), 0)
        prefix = (self.args.output / 'skill_console.log').read_text().strip()
        self.assertEqual(Path(prefix).resolve(), directory.resolve())

    def test_training_uses_hf_runner_without_dcp(self):
        self.args.stage = 'train'
        self.args.dcp = None
        self.args.verified_config = self.root / 'train100.yaml'
        self.args.verified_config.touch()
        plan = migrate.build_plan(self.args)
        self.assertTrue(any(x.endswith('run_hf_training.py') for x in plan['command']))
        self.assertFalse(any(x.endswith('run_mindspeed_qwen35.sh') for x in plan['command']))
        self.assertNotIn('QWEN35_DCP_DIR', plan['environment'])

    def test_training_requires_verified_config(self):
        self.args.stage = 'train'
        with self.assertRaises(ValueError):
            migrate.build_plan(self.args)

    def test_training_needs_confirmation(self):
        self.args.execute = True
        self.args.stage = 'train'
        with self.assertRaises(ValueError):
            migrate.build_plan(self.args)

    def test_existing_output_rejected(self):
        self.args.execute = True
        self.args.output.mkdir()
        with self.assertRaises(ValueError):
            migrate.build_plan(self.args)

    def test_preflight_executes_with_quoted_paths(self):
        plan = migrate.build_plan(self.args)
        self.assertEqual(migrate.execute(plan), 0)
        receipt = json.loads((self.args.output / 'skill_receipt.json').read_text())
        self.assertEqual(receipt['status'], 'command_succeeded')

    def test_failure_retains_evidence(self):
        plan = migrate.build_plan(self.args)
        plan['command'] = [sys.executable, '-c', 'print("failure detail"); raise SystemExit(7)']
        self.assertEqual(migrate.execute(plan), 7)
        receipt = json.loads((self.args.output / 'skill_receipt.json').read_text())
        self.assertEqual(receipt['status'], 'failed')
        self.assertIn('failure detail', (self.args.output / 'skill_console.log').read_text())


if __name__ == '__main__':
    unittest.main()
