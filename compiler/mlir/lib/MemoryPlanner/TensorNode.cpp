#include "MemoryPlanner/TensorNode.h"

TensorNode::TensorNode(mlir::Operation *operation)
    : operation(operation),
      inputs(operation->operand_begin(), operation->operand_end()),
      outputs(operation->result_begin(), operation->result_end()) {}
