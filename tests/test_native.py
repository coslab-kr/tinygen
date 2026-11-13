"""Native integration tests; enable with TINYGEN_TEST_TINYC and TINYGEN_TEST_TRANSLATE."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tinygen import compile_model


TINYC = os.environ.get("TINYGEN_TEST_TINYC")
TRANSLATE = os.environ.get("TINYGEN_TEST_TRANSLATE")


@unittest.skipUnless(TINYC and TRANSLATE, "LLVM/MLIR 20.1 native tools not configured")
class NativeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tinygen-native-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, text, name="model.mlir"):
        path = self.root / name
        path.write_text(text)
        return path

    def native(self, model, *flags):
        return subprocess.run([TINYC, str(model), *flags], cwd=self.root,
                              capture_output=True, text=True)

    @unittest.skipUnless(shutil.which("c++"), "host C++ compiler required")
    def test_float_pipeline_executes_with_separate_input_and_output_buffers(self):
        model = self.write("""module {
  func.func @main(%a: tensor<1x1x1x4xf32>, %b: tensor<1x1x1x4xf32>) -> tensor<1x1x1x4xf32> {
    %0 = tosa.add %a, %b : (tensor<1x1x1x4xf32>, tensor<1x1x1x4xf32>) -> tensor<1x1x1x4xf32>
    return %0 : tensor<1x1x1x4xf32>
  }
}""")
        output = compile_model(model, self.root / "gen", tinyc=TINYC, mlir_translate=TRANSLATE)
        runner = output / "runner.cpp"
        runner.write_text('''#include "model.h"
#include <cmath>
extern "C" { extern Tensor t_arg0; extern Tensor t_arg1; }
int main() {
  model_init();
  float *a = static_cast<float *>(t_arg0.location);
  float *b = static_cast<float *>(t_arg1.location);
  if (a == b) return 1;
  for (int i = 0; i < 4; ++i) { a[i] = i + 0.25f; b[i] = i + 0.5f; }
  Tensor *out = model();
  if (out->location == a || out->location == b) return 2;
  const float *values = static_cast<float *>(out->location);
  for (int i = 0; i < 4; ++i)
    if (std::fabs(values[i] - (2 * i + 0.75f)) > 1e-6f) return 3;
  return 0;
}
''')
        binary = output / "run"
        build = subprocess.run(["c++", "-std=c++17", "-I", str(output),
                                *map(str, sorted(output.glob("*.cpp"))), "-o", str(binary)],
                               capture_output=True, text=True)
        self.assertEqual(build.returncode, 0, build.stderr)
        run = subprocess.run([str(binary)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    @unittest.skipUnless(shutil.which("c++"), "host C++ compiler required")
    def test_constants_broadcast_and_branched_lifetimes(self):
        model = self.write("""module {
  func.func @main(%a: tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32> {
    %c = "tosa.const"() {value = dense<[[[[0.5, 1.5, 2.5]]]]> : tensor<1x1x1x3xf32>} : () -> tensor<1x1x1x3xf32>
    %0 = tosa.add %a, %c : (tensor<1x2x2x3xf32>, tensor<1x1x1x3xf32>) -> tensor<1x2x2x3xf32>
    %1 = tosa.add %a, %0 : (tensor<1x2x2x3xf32>, tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
    %2 = tosa.add %0, %1 : (tensor<1x2x2x3xf32>, tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
    return %2 : tensor<1x2x2x3xf32>
  }
}""")
        output = compile_model(model, self.root / "gen", tinyc=TINYC, mlir_translate=TRANSLATE)
        (output / "runner.cpp").write_text('''#include "model.h"
#include <cmath>
extern "C" { extern Tensor t_arg0; }
int main() {
  model_init();
  auto *input = static_cast<float *>(t_arg0.location);
  for (int i = 0; i < 12; ++i) input[i] = i + 0.25f;
  const auto *output = static_cast<const float *>(model()->location);
  for (int i = 0; i < 12; ++i)
    if (std::fabs(output[i] - (3 * (i + 0.25f) + 2 * (i % 3 + 0.5f))) > 1e-6f) return 1;
  return 0;
}
''')
        binary = output / "run"
        built = subprocess.run(["c++", "-std=c++17", "-I", str(output),
                                *map(str, sorted(output.glob("*.cpp"))), "-o", str(binary)],
                               capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stderr)
        self.assertEqual(subprocess.run([str(binary)]).returncode, 0)

    def test_invalid_main_and_dynamic_shapes_fail_cleanly(self):
        fixtures = [
            ("module {}", "single-block main"),
            ("module { func.func @main() { return } }", "exactly one tensor"),
            ("module { func.func @main(%a: tensor<?xf32>) -> tensor<?xf32> { return %a : tensor<?xf32> } }",
             "static ranked tensors"),
        ]
        for source, expected in fixtures:
            with self.subTest(expected=expected):
                result = self.native(self.write(source))
                self.assertGreater(result.returncode, 0, result.stderr)
                self.assertIn(expected, result.stderr)
                self.assertFalse((self.root / "model_output.mlir").exists())

    def test_dead_rescale_and_integer_add_fail_without_crashing(self):
        sources = [
            """module {
  func.func @main(%a: tensor<4xi8>) -> tensor<4xi8> {
    %0 = tosa.rescale %a {input_zp = 0 : i32, output_zp = 0 : i32,
      multiplier = array<i32: 1073741824>, shift = array<i8: 30>,
      scale32 = true, double_round = false, per_channel = false}
      : (tensor<4xi8>) -> tensor<4xi8>
    return %a : tensor<4xi8>
  }
}""",
            """module {
  func.func @main(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
    %0 = tosa.add %a, %b : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    return %0 : tensor<4xi32>
  }
}""",
        ]
        for source in sources:
            with self.subTest(source=source):
                result = self.native(self.write(source))
                self.assertGreater(result.returncode, 0, result.stderr)
                self.assertIn("failed to legalize", result.stderr)
                self.assertFalse((self.root / "model_output.mlir").exists())

    def test_custom_paths_and_output_errors(self):
        model = self.write("""module {
  func.func @main(%a: tensor<4xi32>) -> tensor<4xi32> { return %a : tensor<4xi32> }
}""")
        result = self.native(model, "-o", "custom.mlir", "--ops-output", "custom.ops")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "custom.mlir").is_file())
        self.assertTrue((self.root / "custom.ops").is_file())
        for flags in (("-o", "missing/out.mlir"), ("--ops-output", "missing/ops.txt")):
            result = self.native(model, *flags)
            self.assertGreater(result.returncode, 0)
            self.assertIn("cannot open", result.stderr)


if __name__ == "__main__":
    unittest.main()
