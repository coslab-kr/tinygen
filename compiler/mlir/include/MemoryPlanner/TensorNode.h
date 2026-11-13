#ifndef TENSOR_NODE_H
#define TENSOR_NODE_H

#include "mlir/IR/Operation.h"
#include <vector>

// Only retain the def/use information required for arena planning.
struct TensorNode {
  explicit TensorNode(mlir::Operation *operation);
  mlir::Operation *operation;
  std::vector<mlir::Value> inputs;
  std::vector<mlir::Value> outputs;
};
#endif
