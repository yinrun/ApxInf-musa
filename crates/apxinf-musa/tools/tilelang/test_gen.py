"""Host-only regression tests for TileLang generation contracts."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("generator", Path(__file__).with_name("gen.py"))
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


class GeneratorTests(unittest.TestCase):
    def test_shared_adapter_source_is_reusable(self):
        source = 'int call() {} int init() {} const char* get_last_error() {}'
        class Adapter:
            def get_kernel_source(self, kernel_only=False):
                self.assert_full = not kernel_only
                return source
        adapter = Adapter()
        obj = types.SimpleNamespace(adapter=adapter)
        self.assertEqual(gen.generated_source(obj), source)
        self.assertEqual(gen.generated_source(obj), source)
        self.assertTrue(adapter.assert_full)

    def test_device_only_source_is_rejected(self):
        obj = types.SimpleNamespace(adapter=types.SimpleNamespace(
            get_kernel_source=lambda **kwargs: '__global__ void kernel() {}'))
        with self.assertRaises(RuntimeError):
            gen.generated_source(obj)

    def test_target_identity_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'op.py'
            path.write_text('OPS = []\n')
            target = dict(arch='mp_22', arch_num='220', mcc='/a/mcc', mcc_sha256='a')
            baseline = gen.op_definition_digest(path, {}, {}, target)
            for key in target:
                changed = dict(target)
                changed[key] += '-changed'
                self.assertNotEqual(baseline, gen.op_definition_digest(path, {}, {}, changed))
            gen.load_module(path)
            self.assertFalse((path.parent / '__pycache__').exists())

    def test_unsupported_target_needs_no_compiler(self):
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, '-B', gen.__file__, '--arch', 'mp_31',
                            '--mcc', '/missing/mcc', '--out', directory], check=True)
            inventory = json.loads((Path(directory) / 'inventory.json').read_text())
            self.assertEqual(inventory['artifacts'], [])


if __name__ == '__main__':
    unittest.main()
