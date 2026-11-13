#ifndef MEMORY_PLANNER_H
#define MEMORY_PLANNER_H

#include "LivenessAnalysis.h"
#include "mlir/Support/LogicalResult.h"
#include <cstddef>
#include <vector>

struct AllocationInfo {
  mlir::Value value;
  TensorNode *defNode = nullptr;
  TensorNode *lastUseNode = nullptr;
  size_t size = 0; // bytes, not tensor elements
  size_t offset = 0;
  bool isModelInput = false;
  bool isModelOutput = false;
};

class MemoryPlanner {
public:
  static constexpr size_t Alignment = 8;
  MemoryPlanner(TensorGraph *graph, LivenessAnalysis *liveness);
  mlir::LogicalResult computeTensorSizes();
  mlir::LogicalResult performMemoryOptimizer();
  size_t getTotalMemoryUsage() const { return totalMemory; }
  size_t getPeakMemoryUsage() const { return peakMemory; }
  const AllocationInfo *getAllocationInfo(mlir::Value value) const;

private:
  LivenessAnalysis *liveness;
  std::vector<AllocationInfo> allocations;
  size_t totalMemory = 0;
  size_t peakMemory = 0;
};
#endif
