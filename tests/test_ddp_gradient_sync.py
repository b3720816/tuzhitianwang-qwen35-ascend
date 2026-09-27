import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class DDP:
    require_backward_grad_sync = True


class FSDP:
    def set_is_last_backward(self, value):
        self.last = value

    def set_requires_all_reduce(self, value):
        self.reduce = value


class GradientSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'third_party/MindSpeed-MM/mindspeed_mm/fsdp/utils/utils.py'
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'configure_hsdp_gradient_sync')
        env = {'torch': SimpleNamespace(nn=SimpleNamespace(parallel=SimpleNamespace(DistributedDataParallel=DDP)))}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), env)
        cls.configure = staticmethod(env[fn.name])

    def test_ddp_two_accumulation_cycles(self):
        model = DDP()
        for value in ([False] * 7 + [True]) * 2:
            self.configure(model, value)
            self.assertIs(model.require_backward_grad_sync, value)

    def test_fsdp_unchanged(self):
        model = FSDP()
        for value in [False, True]:
            self.configure(model, value)
            self.assertIs(model.last, value)
            self.assertIs(model.reduce, value)

    def test_unknown_wrapper_not_silently_accepted(self):
        with self.assertRaises(AttributeError):
            self.configure(object(), False)


if __name__ == '__main__':
    unittest.main()
