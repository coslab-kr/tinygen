#ifndef LIVENESS_ANALYSIS_H
#define LIVENESS_ANALYSIS_H

#include "TensorGraph.h"
#include "llvm/ADT/DenseMap.h"

// The current code generator accepts a single straight-line function.
// Use its execution order directly; another valid topological ordering can
// otherwise release a tensor before its actual emitted last use.
class LivenessAnalysis {
public:
  struct LiveRange {
    TensorNode *defNode = nullptr;
    TensorNode *lastUseNode = nullptr;
    std::vector<TensorNode *> useNodes;
  };

  explicit LivenessAnalysis(TensorGraph *graph);
  int getNodeIndex(TensorNode *node) const;

  llvm::DenseMap<mlir::Value, LiveRange> liveRanges;
  std::vector<TensorNode *> orderedNodes;

private:
  llvm::DenseMap<TensorNode *, int> nodeIndices;
};
#endif
