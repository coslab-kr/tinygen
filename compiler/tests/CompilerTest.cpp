#include "MemoryPlanner/MemoryPlanner.h"
#include "Passes.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include <memory>
#include <string>

using namespace mlir;

namespace {
int failures = 0;
void check(bool condition, const char *message) {
  if (!condition) {
    llvm::errs() << "FAIL: " << message << '\n';
    ++failures;
  }
}

void testPlanner(MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%a: tensor<4xf32>, %b: tensor<4xi32>) -> tensor<4xf32> {
        %0 = "test.op"(%a) : (tensor<4xf32>) -> tensor<4xf32>
        %1 = "test.op"(%0, %b) : (tensor<4xf32>, tensor<4xi32>) -> tensor<4xf32>
        %2 = "test.op"(%1) : (tensor<4xf32>) -> tensor<4xf32>
        return %2 : tensor<4xf32>
      }
    })mlir",
                                            &context);
  check(bool(module), "planner fixture parses");
  if (!module)
    return;
  auto function = module->lookupSymbol<func::FuncOp>("main");
  TensorGraph graph;
  auto *inputs = graph.addNode(function);
  for (Value arg : function.getArguments())
    inputs->outputs.push_back(arg);
  for (Operation &op : function.front())
    graph.addNode(&op);
  LivenessAnalysis liveness(&graph);
  MemoryPlanner planner(&graph, &liveness);
  check(succeeded(planner.computeTensorSizes()), "tensor byte sizes are valid");
  check(succeeded(planner.performMemoryOptimizer()), "allocation succeeds");
  auto first = function.front().begin();
  Value a = function.getArgument(0), b = function.getArgument(1);
  Value x = first++->getResult(0), y = first++->getResult(0),
        z = first->getResult(0);
  const auto *aa = planner.getAllocationInfo(a),
             *bb = planner.getAllocationInfo(b);
  const auto *xx = planner.getAllocationInfo(x),
             *yy = planner.getAllocationInfo(y);
  const auto *zz = planner.getAllocationInfo(z);
  check(aa && bb && xx && yy && zz, "every input and result has storage");
  if (!aa || !bb || !xx || !yy || !zz)
    return;
  check(aa->size == 16 && bb->size == 16, "float32 and int32 sizes use bytes");
  check(aa->offset != bb->offset, "model inputs have distinct buffers");
  check(xx->offset != yy->offset,
        "last-use inputs cannot alias current outputs");
  check(zz->offset == xx->offset, "expired intermediate storage is reused");
  check(zz->offset != aa->offset && zz->offset != bb->offset,
        "output cannot alias pinned inputs");
  check(planner.getPeakMemoryUsage() == 64, "arena tracks actual aligned peak");
  check(succeeded(planner.computeTensorSizes()) &&
            succeeded(planner.performMemoryOptimizer()),
        "planning can be repeated without accumulating allocations");
  check(planner.getPeakMemoryUsage() == 64, "repeated planning is stable");
}

void testRepeatedPass(MLIRContext &context) {
  int fd;
  llvm::SmallString<128> path;
  check(!llvm::sys::fs::createTemporaryFile("tinygen-ops", "txt", fd, path),
        "temporary operator file");
  llvm::sys::fs::closeFile(fd);
  PassManager manager(&context);
  manager.addPass(tosa::createTosaToEmitC(path.str().str()));
  std::string previous;
  for (int i = 0; i < 2; ++i) {
    auto module = parseSourceString<ModuleOp>(R"mlir(
      module {
        func.func @main(%a: tensor<1x1x1x4xf32>, %b: tensor<1x1x1x4xf32>) -> tensor<1x1x1x4xf32> {
          %0 = tosa.add %a, %b : (tensor<1x1x1x4xf32>, tensor<1x1x1x4xf32>) -> tensor<1x1x1x4xf32>
          return %0 : tensor<1x1x1x4xf32>
        }
      })mlir",
                                              &context);
    check(bool(module), "lowering fixture parses");
    if (!module)
      break;
    check(succeeded(manager.run(*module)), "native lowering succeeds");
    check(succeeded(verify(*module)), "lowered module verifies");
    std::string text;
    llvm::raw_string_ostream stream(text);
    module->print(stream);
    if (i)
      check(text == previous,
            "repeated passes do not leak names or MLIR handles");
    previous = std::move(text);
  }
  llvm::sys::fs::remove(path);
}
} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<func::FuncDialect, tosa::TosaDialect, emitc::EmitCDialect>();
  MLIRContext context(registry);
  context.allowUnregisteredDialects();
  testPlanner(context);
  testRepeatedPass(context);
  return failures ? 1 : 0;
}
