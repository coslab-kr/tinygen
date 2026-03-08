#include "MemoryPlanner/LivenessAnalysis.h"

LivenessAnalysis::LivenessAnalysis(TensorGraph *graph) {
    // 1. Topologically sort the graph nodes
    topologicalSort(graph);
    
    // 2. Find defining and using nodes for each value
    buildDefUseInfo(graph);
    
    // 3. liveness analysis in reverse topological order
    computeLiveness();
    
    // 4. Compute live ranges for each value
    computeLiveRanges();
}

// Topologically sort the graph nodes (DFS base)
void LivenessAnalysis::topologicalSort(TensorGraph* graph) {
    std::set<TensorNode*> visited;
    std::set<TensorNode*> temp; // to check cycle
    
    for (auto node : graph->nodes) {
        if (visited.find(node) == visited.end()) {
            topoSortUtil(node, visited, temp, graph);
        }
    }
}

void LivenessAnalysis::topoSortUtil(TensorNode* node, std::set<TensorNode*>& visited, 
                    std::set<TensorNode*>& temp, TensorGraph* graph) {
    if (temp.find(node) != temp.end()) {
        llvm::errs() << "Error: Cycle detected in graph\n";
        return;
    }
    
    if (visited.find(node) != visited.end()) {
        return;
    }
    
    temp.insert(node);
    
    for (auto output : node->outputs) {
        auto users = graph->userNodes.find(output);
        if (users != graph->userNodes.end()) {
            for (auto userNode : users->second) {
                topoSortUtil(userNode, visited, temp, graph);
            }
        }
    }
    
    temp.erase(node);
    visited.insert(node);
    
    topoSortedNodes.insert(topoSortedNodes.begin(), node);
}

void LivenessAnalysis::buildDefUseInfo(TensorGraph* graph) {
    for (auto node : graph->nodes) {
        for (auto output : node->outputs) {
            liveRanges[output].defNode = node;
        }
        
        for (auto input : node->inputs) {
            if (graph->definingNodes.find(input) != graph->definingNodes.end()) {
                auto defNode = graph->definingNodes[input];
                liveRanges[input].useNodes.push_back(node);
            }
        }
    }
}

void LivenessAnalysis::computeLiveness() {
    bool changed = true;
    
    for (auto node : topoSortedNodes) {
        liveIn[node] = llvm::DenseSet<mlir::Value>();
        liveOut[node] = llvm::DenseSet<mlir::Value>();
    }
    
    while (changed) {
        changed = false;
        
        for (auto it = topoSortedNodes.rbegin(); it != topoSortedNodes.rend(); ++it) {
            TensorNode* node = *it;
            
            size_t oldInSize = liveIn[node].size();
            size_t oldOutSize = liveOut[node].size();
            
            for (auto output : node->outputs) {
                for (auto user : liveRanges[output].useNodes) {
                    for (auto val : liveIn[user]) {
                        liveOut[node].insert(val);
                    }
                }
            }
            
            for (auto input : node->inputs) {
                liveIn[node].insert(input);
            }
            
            for (auto val : liveOut[node]) {
                bool isDefined = false;
                for (auto output : node->outputs) {
                    if (val == output) {
                        isDefined = true;
                        break;
                    }
                }
                
                if (!isDefined) {
                    liveIn[node].insert(val);
                }
            }
            
            if (liveIn[node].size() != oldInSize || liveOut[node].size() != oldOutSize) {
                changed = true;
            }
        }
    }
}

void LivenessAnalysis::computeLiveRanges() {
    for (auto& entry : liveRanges) {
        mlir::Value val = entry.first;
        LiveRange& range = entry.second;
        
        if (!range.useNodes.empty()) {
            range.lastUseNode = range.useNodes[0];
            int lastUseIdx = -1;
            
            for (auto useNode : range.useNodes) {
                int currentIdx = -1;
                for (size_t i = 0; i < topoSortedNodes.size(); i++) {
                    if (topoSortedNodes[i] == useNode) {
                        currentIdx = i;
                        break;
                    }
                }
                
                if (currentIdx > lastUseIdx) {
                    lastUseIdx = currentIdx;
                    range.lastUseNode = useNode;
                }
            }
        } else {
            range.lastUseNode = nullptr;
        }
    }
}

