#include "MemoryPlanner/MemoryPlanner.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>

MemoryPlanner::MemoryPlanner(TensorGraph* graph, LivenessAnalysis* liveness)
    : graph(graph), liveness(liveness), totalMemory(0), peakMemory(0) {
}

MemoryPlanner::~MemoryPlanner() {
    // No manual cleanup needed - using STL containers
}

void MemoryPlanner::computeTensorSizes() {
    // Process all values in the liveness analysis
    for (auto& entry : liveness->liveRanges) {
        mlir::Value &val = entry.getFirst();
        auto& range = entry.getSecond();
        
        // Skip non-tensor values
        if (!mlir::isa<mlir::ShapedType>(val.getType()))
            continue;
            
        // Create allocation info
        AllocationInfo alloc;
        alloc.value = &val;
        alloc.defNode = range.defNode;
        alloc.lastUseNode = range.lastUseNode;
        alloc.size = estimateTensorSize(&val);
        alloc.allocatedPoolIndex = -1;
        alloc.offset = -1;
        
        // Check if this is an input or output
        alloc.isModelInput = (range.defNode == nullptr); // No defining node means it's an input
        alloc.isModelOutput = true; // Assume it's an output by default
        
        // If this value is used by any node, it's not an output
        if (!range.useNodes.empty()) {
            alloc.isModelOutput = false;
        }
        
        // Special case: if it's the result of the last node and has no uses, it's an output
        if (range.defNode && range.useNodes.empty()) {
            alloc.isModelOutput = true;
        }
        
        // Add to our allocation list
        allocations.push_back(alloc);
        
        // Update total memory
        totalMemory += alloc.size;
    }
    
}

size_t MemoryPlanner::estimateTensorSize(mlir::Value *tensor) {
    // Get the tensor type
    auto type =  mlir::dyn_cast<mlir::ShapedType>(tensor->getType());
    if (!type) {
        return 0; // Not a tensor
    }
    
    // Calculate number of elements
    int64_t numElements = 1;
    for (auto dim : type.getShape()) {
        if (dim < 0) {
            // Dynamic dimension, use a default size
            dim = 1;
        }
        numElements *= dim;
    }
    return numElements;
}

void MemoryPlanner::buildAllocationPlan() {
    
    llvm::outs() << "Building memory allocation plan...\n";

    std::sort(allocations.begin(), allocations.end(),
        [&](const AllocationInfo& a, const AllocationInfo& b) {
            if (a.isModelInput && !b.isModelInput) return true;
            if (!a.isModelInput && b.isModelInput) return false;
            if (a.defNode && b.defNode) {
                int aIdx = -1, bIdx = -1;
                for (size_t i = 0; i < liveness->topoSortedNodes.size(); i++) {
                    if (liveness->topoSortedNodes[i] == a.defNode) aIdx = i;
                    if (liveness->topoSortedNodes[i] == b.defNode) bIdx = i;
                }
                return aIdx < bIdx;
            }
            return false;
        });

    size_t currentOffset = 0;

    for (auto &alloc : allocations) {
        if (alloc.isModelInput || alloc.isModelOutput) {
            alloc.allocatedPoolIndex = 0;
            alloc.offset = 0;
        } else {
            alloc.allocatedPoolIndex = 0; 
            alloc.offset = currentOffset;
            currentOffset += alloc.size;
        }
    }

    peakMemory = currentOffset;
    totalMemory = currentOffset;

    llvm::outs() << "Memory allocation plan complete.\n";
    llvm::outs() << "Total memory: " << (totalMemory / 1024.0) << " KB\n";
    llvm::outs() << "Peak memory: " << (peakMemory / 1024.0) << " KB\n";
    llvm::outs() << "Memory efficiency: " << ((float)totalMemory / peakMemory) * 100.0f << "%\n";
}

int MemoryPlanner::findMemoryPool(AllocationInfo& alloc) {
    // Try to find a memory pool that has enough free space
    for (size_t i = 0; i < memoryPools.size(); i++) {
        auto& pool = memoryPools[i];
        
        // Check each free interval
        for (auto& interval : pool.freeIntervals) {
            size_t intervalSize = interval.second - interval.first;
            if (intervalSize >= alloc.size) {
                return i; // Found a suitable pool
            }
        }
    }
    
    return -1; // No suitable pool found
}

void MemoryPlanner::insertIntoMemoryPool(int poolIndex, AllocationInfo& alloc) {
    auto& pool = memoryPools[poolIndex];
    
    // Find a free interval that's large enough
    for (auto it = pool.freeIntervals.begin(); it != pool.freeIntervals.end(); ++it) {
        size_t intervalSize = it->second - it->first;
        if (intervalSize >= alloc.size) {
            // Found a suitable interval
            alloc.allocatedPoolIndex = poolIndex;
            alloc.offset = it->first;
            
            // Update the free interval
            int newStart = it->first + alloc.size;
            int oldEnd = it->second;
            
            // Remove this interval
            it = pool.freeIntervals.erase(it);
            
            // If there's still free space after this allocation, add a new interval
            if (newStart < oldEnd) {
                pool.freeIntervals.push_back(std::make_pair(newStart, oldEnd));
            }
            
            // Sort free intervals by start offset
            std::sort(pool.freeIntervals.begin(), pool.freeIntervals.end(),
                [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                    return a.first < b.first;
                });
            
            return;
        }
    }
    
    // This should not happen if findMemoryPool returned a valid pool
    llvm::errs() << "Error: Failed to insert tensor into memory pool\n";
}

int MemoryPlanner::createNewMemoryPool(size_t initialSize) {
    MemoryPool pool;
    pool.size = initialSize;
    pool.name = "pool_" + std::to_string(memoryPools.size());
    
    // Initially, the entire pool is free
    pool.freeIntervals.push_back(std::make_pair(0, initialSize));
    
    memoryPools.push_back(pool);
    return memoryPools.size() - 1;
}

void MemoryPlanner::performMemoryOptimizer() {

    memoryPools.clear();
    int poolIdx = createNewMemoryPool(1024 * 1024); 
    auto& pool = memoryPools[poolIdx];

    pool.freeIntervals.clear();
    pool.freeIntervals.push_back({0, pool.size});

    std::sort(allocations.begin(), allocations.end(),
        [&](const AllocationInfo& a, const AllocationInfo& b) {
            int aIdx = getNodeIndex(a.defNode);
            int bIdx = getNodeIndex(b.defNode);
            return aIdx < bIdx;
        });

    for (auto& alloc : allocations) {
        if (alloc.isModelInput || alloc.isModelOutput) {
            alloc.allocatedPoolIndex = 0;
            alloc.offset = 0;
            continue;
        }

        int defIdx = getNodeIndex(alloc.defNode);
        bool placed = false;

        for (auto it = pool.freeIntervals.begin(); it != pool.freeIntervals.end(); ++it) {
            size_t freeSize = it->second - it->first;
            if (freeSize >= alloc.size) {
                alloc.allocatedPoolIndex = 0;
                alloc.offset = it->first;

                size_t newStart = it->first + alloc.size;
                size_t oldEnd = it->second;
                pool.freeIntervals.erase(it);

                if (newStart < oldEnd)
                    pool.freeIntervals.push_back({newStart, oldEnd});

                placed = true;
                break;
            }
        }

        if (!placed) {
            size_t oldSize = pool.size;
            size_t expandSize = std::max(alloc.size, (size_t)64 * 1024);
            pool.size += expandSize;

            alloc.allocatedPoolIndex = 0;
            alloc.offset = oldSize;

            if (alloc.size < expandSize)
                pool.freeIntervals.push_back({oldSize + alloc.size, pool.size});

            llvm::outs() << "  [Expand] Arena expanded by " << (expandSize / 1024.0)
                         << " KB for tensor (" << (alloc.size / 1024.0) << " KB)\n";
        }

        for (auto& prevAlloc : allocations) {
            if (prevAlloc.allocatedPoolIndex != 0) continue;
            if (prevAlloc.offset < 0) continue;

            int prevLast = getNodeIndex(prevAlloc.lastUseNode);
            if (prevLast < defIdx) {
                pool.freeIntervals.push_back({prevAlloc.offset,
                                              prevAlloc.offset + prevAlloc.size});
                prevAlloc.allocatedPoolIndex = -1;
            }
        }

        compactMemoryPools();
    }

    size_t memory_size = 0;
    for (const auto& alloc : allocations) {
        if (alloc.offset >= 0 && alloc.size > 0) {
            size_t endAddr = alloc.offset + alloc.size;
            if (endAddr > memory_size)
                memory_size = endAddr;
        }
    }

    memory_size += 1024;

    peakMemory = memory_size;
    totalMemory = memory_size;
    pool.size = memory_size;

}

// Helper method to get node index in topological order
int MemoryPlanner::getNodeIndex(TensorNode* node) {
    if (!node) return liveness->topoSortedNodes.size(); // Default to end of execution
    
    for (size_t i = 0; i < liveness->topoSortedNodes.size(); i++) {
        if (liveness->topoSortedNodes[i] == node) return i;
    }
    
    return -1; // Not found (shouldn't happen)
}

void MemoryPlanner::compactMemoryPools() {
    for (auto& pool : memoryPools) {
        if (pool.freeIntervals.empty()) continue;
        
        // Sort by start offset
        std::sort(pool.freeIntervals.begin(), pool.freeIntervals.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                return a.first < b.first;
            });
        
        // Merge adjacent intervals
        std::vector<std::pair<int, int>> mergedIntervals;
        mergedIntervals.push_back(pool.freeIntervals[0]);
        
        for (size_t i = 1; i < pool.freeIntervals.size(); i++) {
            auto& last = mergedIntervals.back();
            auto& current = pool.freeIntervals[i];
            
            if (last.second == current.first) {
                // Adjacent intervals, merge them
                last.second = current.second;
            } else {
                // Non-adjacent, add as a new interval
                mergedIntervals.push_back(current);
            }
        }
        
        pool.freeIntervals = mergedIntervals;
    }
}

size_t MemoryPlanner::getTotalMemoryUsage() const {
    return totalMemory;
}

size_t MemoryPlanner::getPeakMemoryUsage() const {
    return peakMemory;
}

const AllocationInfo *MemoryPlanner::getAllocationInfo(mlir::Value value) const{
    for(const auto& alloc : allocations){
        if(*(alloc.value) == value){
            return &alloc;
        }
    }
    return nullptr;
}


