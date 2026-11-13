#ifndef TENSOR_GRAPH_H
#define TENSOR_GRAPH_H

#include "TensorNode.h"
#include <memory>
#include <vector>

class TensorGraph {
public:
  TensorNode *addNode(mlir::Operation *operation);
  std::vector<std::unique_ptr<TensorNode>> nodes;
};
#endif
