#include "MemoryPlanner/TensorGraph.h"

TensorNode *TensorGraph::addNode(mlir::Operation *operation) {
  nodes.push_back(std::make_unique<TensorNode>(operation));
  return nodes.back().get();
}
