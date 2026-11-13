#include "Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

namespace cl = llvm::cl;
namespace {
cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input TOSA MLIR>"), cl::init("-"));
cl::opt<std::string> outputFilename("o", cl::desc("EmitC output file"),
                                    cl::init("model_output.mlir"));
cl::opt<std::string> opsFilename("ops-output",
                                 cl::desc("Operator list output file"),
                                 cl::init("model_ops_list.txt"));
} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "TinyGen TOSA compiler\n");
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::emitc::EmitCDialect,
                  mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                  mlir::quant::QuantDialect, mlir::tensor::TensorDialect,
                  mlir::tosa::TosaDialect>();
  mlir::MLIRContext context(registry);
  auto input = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!input) {
    llvm::errs() << "tinyc: cannot read input: " << input.getError().message()
                 << '\n';
    return 1;
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*input), llvm::SMLoc());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module)
    return 1;

  mlir::PassManager manager(&context);
  manager.addPass(mlir::tosa::createTosaToEmitC(opsFilename));
  if (mlir::failed(manager.run(*module)))
    return 1;

  std::error_code error;
  llvm::ToolOutputFile output(outputFilename, error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << "tinyc: cannot open output: " << error.message() << '\n';
    return 1;
  }
  module->print(output.os());
  output.os() << '\n';
  output.os().flush();
  if (output.os().has_error()) {
    llvm::errs() << "tinyc: cannot write output: "
                 << output.os().error().message() << '\n';
    return 1;
  }
  output.keep();
  return 0;
}
