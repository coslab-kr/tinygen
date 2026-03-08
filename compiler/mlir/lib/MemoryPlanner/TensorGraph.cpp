#include "MemoryPlanner/TensorGraph.h"

// constructor
TensorGraph::TensorGraph() {
}

// destructor
TensorGraph::~TensorGraph() {
    for (auto node : nodes) {
        delete node;
    }
}
