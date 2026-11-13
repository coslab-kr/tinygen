from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tinygen.api import _assets
from tinygen._kernels import collect_kernels


class KernelsTest(unittest.TestCase):
    def run_compiler(self, command):
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which("cc") and shutil.which("c++"), "C/C++ compilers required")
    def test_all_cmsis_adapters_compile_and_link_on_host(self):
        """Catch changed upstream APIs and omitted transitive source dependencies.

        This checks scalar host compilation/linking, not Cortex-M execution.
        """
        kernels, cmsis = _assets()
        with tempfile.TemporaryDirectory(prefix="tinygen-kernels-") as tmp:
            output = Path(tmp)
            ops = sorted(p.stem for p in (kernels / "cmsis-nn").glob("*.cpp"))
            op_file = output / "model_ops_list.txt"
            op_file.write_text("\n".join(ops) + "\n")
            collect_kernels(op_file, output, kernels, cmsis)
            self.assertTrue((output / "arm_nn_mat_mult_nt_t_s8.c").is_file())
            objects = []
            for source in sorted(output.glob("*.c")) + sorted(output.glob("*.cpp")):
                target = source.with_suffix(source.suffix + ".o")
                compiler = "cc" if source.suffix == ".c" else "c++"
                self.run_compiler([compiler, "-I", str(output), "-c", str(source),
                                   "-o", str(target)])
                objects.append(str(target))
            main = output / "main.cpp"
            main.write_text("int main() { return 0; }\n")
            self.run_compiler(["c++", str(main), *objects, "-o", str(output / "check")])

    def test_missing_upstream_symbol_fails_instead_of_silently_omitting_it(self):
        kernels, cmsis = _assets()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cmsis-nn").mkdir()
            (root / "cmsis-nn/new_op.cpp").write_text("void f() { arm_missing_function(); }")
            ops = root / "ops.txt"
            ops.write_text("new_op\n")
            with self.assertRaisesRegex(RuntimeError, "arm_missing_function"):
                collect_kernels(ops, root / "out", root, cmsis)


if __name__ == "__main__":
    unittest.main()
