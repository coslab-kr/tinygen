#include "MemoryPlanner/LivenessAnalysis.h"

LivenessAnalysis::LivenessAnalysis(TensorGraph *graph) {
  for (const auto &node : graph->nodes)
    orderedNodes.push_back(node.get());
  for (size_t i = 0; i < orderedNodes.size(); ++i) {
    TensorNode *node = orderedNodes[i];
    nodeIndices[node] = static_cast<int>(i);
    for (mlir::Value output : node->outputs) {
      auto &range = liveRanges[output];
      range.defNode = node;
      range.lastUseNode = node;
    }
  }
  for (TensorNode *node : orderedNodes) {
    for (mlir::Value input : node->inputs) {
      auto found = liveRanges.find(input);
      if (found == liveRanges.end())
        continue; // Constants have separate immutable storage.
      auto &range = found->second;
      range.useNodes.push_back(node);
      range.lastUseNode = node;
    }
  }
}

int LivenessAnalysis::getNodeIndex(TensorNode *node) const {
  auto found = nodeIndices.find(node);
  return found == nodeIndices.end() ? -1 : found->second;
}
