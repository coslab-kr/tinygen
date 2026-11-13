#include "MemoryPlanner/MemoryPlanner.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include <algorithm>
#include <limits>

using namespace mlir;

namespace {
// The generated arena and offsets use signed 64-bit EmitC integer attributes.
constexpr size_t MaxSize =
    static_cast<size_t>(std::numeric_limits<int64_t>::max()) - 8;

FailureOr<size_t> tensorBytes(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape())
    return failure();
  Type element = type.getElementType();
  if (auto quant = dyn_cast<quant::QuantizedType>(element))
    element = quant.getStorageType();
  if (!element.isIntOrFloat())
    return failure();
  size_t bytes = (element.getIntOrFloatBitWidth() + 7) / 8;
  if (bytes == 0)
    return failure();
  for (int64_t dim : type.getShape()) {
    if (dim <= 0 || static_cast<uint64_t>(dim) > MaxSize / bytes)
      return failure();
    bytes *= static_cast<size_t>(dim);
  }
  return bytes;
}

size_t alignSize(size_t size) {
  return (size + MemoryPlanner::Alignment - 1) &
         ~(MemoryPlanner::Alignment - 1);
}

struct Interval {
  size_t begin;
  size_t end;
};

void mergeIntervals(std::vector<Interval> &intervals) {
  std::sort(intervals.begin(), intervals.end(),
            [](Interval a, Interval b) { return a.begin < b.begin; });
  std::vector<Interval> merged;
  for (Interval interval : intervals) {
    if (!merged.empty() && interval.begin <= merged.back().end)
      merged.back().end = std::max(merged.back().end, interval.end);
    else
      merged.push_back(interval);
  }
  intervals = std::move(merged);
}
} // namespace

MemoryPlanner::MemoryPlanner(TensorGraph *, LivenessAnalysis *liveness)
    : liveness(liveness) {}

LogicalResult MemoryPlanner::computeTensorSizes() {
  allocations.clear();
  totalMemory = peakMemory = 0;
  // Iterating in program order also gives deterministic offsets for equal
  // lifetimes.
  for (TensorNode *node : liveness->orderedNodes) {
    for (Value value : node->outputs) {
      if (!isa<ShapedType>(value.getType()))
        continue;
      auto bytes = tensorBytes(value);
      if (failed(bytes))
        return emitError(value.getLoc(),
                         "cannot allocate tensor with dynamic, empty, "
                         "unsupported or overflowing shape");
      const auto &range = liveness->liveRanges.find(value)->second;
      AllocationInfo alloc;
      alloc.value = value;
      alloc.defNode = range.defNode;
      alloc.lastUseNode = range.lastUseNode;
      alloc.size = *bytes;
      alloc.isModelInput = isa<BlockArgument>(value);
      alloc.isModelOutput = std::any_of(
          range.useNodes.begin(), range.useNodes.end(), [](TensorNode *user) {
            return isa<func::ReturnOp>(user->operation);
          });
      allocations.push_back(alloc);
    }
  }
  return success();
}

LogicalResult MemoryPlanner::performMemoryOptimizer() {
  struct Active {
    size_t allocation;
    size_t end;
  };
  std::vector<Interval> free;
  std::vector<Active> active;
  size_t highWater = 0;
  for (size_t i = 0; i < allocations.size(); ++i) {
    auto &alloc = allocations[i];
    int definition = liveness->getNodeIndex(alloc.defNode);
    // Free before allocating, exactly once. Strict inequality prevents an
    // operator's output from overwriting its still-needed input at that
    // operator.
    for (auto it = active.begin(); it != active.end();) {
      const auto &previous = allocations[it->allocation];
      if (!previous.isModelInput && !previous.isModelOutput &&
          liveness->getNodeIndex(previous.lastUseNode) < definition) {
        free.push_back({previous.offset, it->end});
        it = active.erase(it);
      } else {
        ++it;
      }
    }
    mergeIntervals(free);
    size_t bytes = alignSize(alloc.size);
    auto slot =
        std::find_if(free.begin(), free.end(), [bytes](Interval interval) {
          return interval.end - interval.begin >= bytes;
        });
    if (slot != free.end()) {
      alloc.offset = slot->begin;
      slot->begin += bytes;
      if (slot->begin == slot->end)
        free.erase(slot);
    } else {
      if (bytes > MaxSize - highWater)
        return emitError(alloc.value.getLoc(),
                         "tensor arena exceeds supported address range");
      alloc.offset = highWater;
      highWater += bytes;
    }
    active.push_back({i, alloc.offset + bytes});
  }
  totalMemory = peakMemory = std::max(highWater, Alignment);
  return success();
}

const AllocationInfo *MemoryPlanner::getAllocationInfo(Value value) const {
  auto found = std::find_if(
      allocations.begin(), allocations.end(),
      [value](const AllocationInfo &alloc) { return alloc.value == value; });
  return found == allocations.end() ? nullptr : &*found;
}
