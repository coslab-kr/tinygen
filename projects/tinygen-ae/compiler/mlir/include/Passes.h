#ifndef TINYGEN_PASSES_H
#define TINYGEN_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace tosa {
std::unique_ptr<Pass> createTosaToEmitC();
} // namespace tosa

} // namespace mlir

#endif // TINYGEN_PASSES_H
