"""CPU-only configuration regression tests; these do not prove NPU execution.

Run from the repository root with Python and PyYAML installed:
    python -m unittest discover -s tests -v
"""

import importlib.util
from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "config_builder", ROOT / "scripts/prepare_mindspeed_qwen35_config.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ReproductionConfigTest(unittest.TestCase):
    def setUp(self):
        self.template = yaml.safe_load(builder.DEFAULT_TEMPLATE.read_text())
        self.expected = yaml.safe_load(
            (ROOT / "verified_run/successful_run.yaml").read_text()
        )
        basic = self.expected["data"]["dataset_param"]["basic_parameters"]
        self.args = dict(
            hf_model_dir=Path(self.expected["model"]["model_name_or_path"]),
            dcp_dir=Path(self.expected["training"]["load"]),
            dataset_file=Path(basic["dataset"]),
            cache_dir=Path(basic["cache_dir"]),
            save_dir=Path(self.expected["training"]["save"]),
            train_iters=100,
            num_workers=0,
        )

    def test_matches_complete_successful_config(self):
        self.assertEqual(builder.build_config(self.template, **self.args), self.expected)

    def test_template_not_mutated(self):
        original = yaml.safe_dump(self.template)
        builder.build_config(self.template, **self.args)
        self.assertEqual(yaml.safe_dump(self.template), original)

    def test_explicit_override(self):
        self.args["num_workers"] = 2
        result = builder.build_config(
            self.template, preprocessing_num_workers=3, **self.args
        )
        self.assertEqual(builder.config_contract(result)["num_workers"], 2)
        self.assertEqual(builder.config_contract(result)["preprocessing_num_workers"], 3)

    def test_reject_invalid_workers(self):
        with self.assertRaises(ValueError):
            builder.build_config(self.template, preprocessing_num_workers=0, **self.args)
        self.args["num_workers"] = -1
        with self.assertRaises(ValueError):
            builder.build_config(self.template, **self.args)


if __name__ == "__main__":
    unittest.main()
