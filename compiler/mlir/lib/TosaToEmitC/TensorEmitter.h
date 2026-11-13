#ifndef TINYGEN_TENSOR_EMITTER_H
#define TINYGEN_TENSOR_EMITTER_H
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace tinygen {
mlir::Value globalArrayAddress(mlir::OpBuilder &builder,
                               mlir::Location location, llvm::StringRef name,
                               mlir::Type element,
                               llvm::ArrayRef<int64_t> shape, mlir::Value zero);
void initializeTensor(mlir::OpBuilder &builder, mlir::Location location,
                      llvm::StringRef name, mlir::RankedTensorType type,
                      mlir::Value data, mlir::Value zero);
} // namespace tinygen
#endif
