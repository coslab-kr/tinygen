#ifndef TINYGEN_PASSES_H
#define TINYGEN_PASSES_H

#include <memory>
#include <string>

namespace mlir {
class Pass;

namespace tosa {
std::unique_ptr<Pass>
createTosaToEmitC(std::string opsFile = "model_ops_list.txt");
} // namespace tosa

} // namespace mlir

#endif // TINYGEN_PASSES_H
