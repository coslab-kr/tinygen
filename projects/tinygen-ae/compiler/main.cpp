#include <iostream>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "Passes.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"


namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input tosa file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

void saveModuleToFile(mlir::ModuleOp module, const std::string &fileName) {
    std::error_code errorCode;
    llvm::raw_fd_ostream outFile(fileName, errorCode, llvm::sys::fs::OF_None);

    if (errorCode) {
        std::cerr << "Failed to open file: " << fileName << " (" << errorCode.message() << ")" << std::endl;
        return;
    }

    module.print(outFile);
}


int main (int argc, char* argv[]) {
  cl::ParseCommandLineOptions(argc, argv, "tinyc compiler\n");

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::tosa::TosaDialect>();
  context.getOrLoadDialect<mlir::emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);

  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }

  mlir::PassManager pm(&context);
  pm.addPass(mlir::tosa::createTosaToEmitC());

  if (mlir::failed(pm.run(*module)))
    return 4;

  mlir::ModuleOp actualModule = module.get();
  saveModuleToFile(actualModule, "model_output.mlir");
  return 0;
}
