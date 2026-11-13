import concurrent.futures
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tinygen import compile_model, lower_model
from tinygen.cli import main


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tinygen test ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "test model.mlir"
        self.model.write_text("module {}")
        self.compiler = self.executable("tinyc", """
from pathlib import Path
import sys
assert Path(sys.argv[1]).is_absolute()
Path('model_output.mlir').write_text('module {}')
Path('model_ops_list.txt').write_text('add_s8\\ntranspose\\n')
""")
        self.translator = self.executable("mlir-translate", """
from pathlib import Path
import sys
assert sys.argv[1] == '--mlir-to-cpp'
assert Path(sys.argv[2]).is_file()
print('// translated C++')
""")

    def executable(self, name, body):
        path = self.root / name
        path.write_text(f"#!{sys.executable}\n" + body)
        path.chmod(0o755)
        return path

    def test_full_pipeline_and_cwd_independence(self):
        cwd = Path.cwd()
        result = compile_model(self.model, self.root / "out", tinyc=self.compiler,
                               mlir_translate=self.translator)
        self.assertEqual(result.name, "test model_tinygen")
        self.assertIn("translated C++", (result / "test model.cpp").read_text())
        self.assertIn("#define ADD_S8", (result / "kernels.h").read_text())
        self.assertTrue((result / "arm_elementwise_add_s8.c").is_file())
        self.assertTrue((result / "arm_transpose_s8.c").is_file())
        self.assertTrue((result / "Internal/arm_nn_compiler.h").is_file())
        self.assertTrue((result / "CMSIS-NN-LICENSE.txt").is_file())
        self.assertEqual(Path.cwd(), cwd)
        self.assertFalse((self.root / "model_output.mlir").exists())
        with self.assertRaises(FileExistsError):
            compile_model(self.model, self.root / "out", tinyc=self.compiler,
                          mlir_translate=self.translator)

    def test_lower_only_and_cli(self):
        result = lower_model(self.model, self.root / "lower", tinyc=self.compiler)
        self.assertEqual(result.read_text(), "module {}")
        self.assertEqual(main([str(self.model), "--emitc-only", "--tinyc",
                               str(self.compiler), "-o", str(self.root / "cli")]), 0)

    def test_failure_is_propagated_without_partial_output(self):
        broken = self.executable("broken", "raise SystemExit(7)\n")
        for compiler, translator in ((broken, self.translator), (self.compiler, broken)):
            with self.assertRaises(subprocess.CalledProcessError) as failure:
                compile_model(self.model, self.root / "failed", tinyc=compiler,
                              mlir_translate=translator)
            self.assertEqual(failure.exception.returncode, 7)
            self.assertEqual(list((self.root / "failed").iterdir()), [])

    def test_missing_inputs_and_explicit_executable(self):
        with self.assertRaises(FileNotFoundError):
            lower_model(self.root / "missing.mlir", tinyc=self.compiler)
        with self.assertRaises(FileNotFoundError):
            lower_model(self.model, tinyc=self.root / "missing-tool")
        incomplete = self.executable("incomplete", "pass\n")
        with self.assertRaisesRegex(RuntimeError, "did not produce"):
            lower_model(self.model, self.root / "incomplete-out", tinyc=incomplete)

    def test_concurrent_calls_use_separate_workspaces(self):
        def generate(index):
            return compile_model(self.model, self.root / str(index), tinyc=self.compiler,
                                 mlir_translate=self.translator)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(generate, range(2)))
        self.assertTrue(all((p / "model_ops_list.txt").is_file() for p in results))


if __name__ == "__main__":
    unittest.main()
