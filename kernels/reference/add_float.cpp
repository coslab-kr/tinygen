#include "kernels.h"
#include "reference.h"

#ifdef ADD_FLOAT
namespace {
// Map an output element to a right-aligned, possibly broadcast input tensor.
size_t broadcastOffset(size_t element, const Tensor *input,
                       const Tensor *output) {
  size_t offset = 0;
  size_t stride = 1;
  for (int axis = output->rank - 1; axis >= 0; --axis) {
    const size_t coordinate = element % output->shape[axis];
    element /= output->shape[axis];
    const int inputAxis = axis - (output->rank - input->rank);
    if (inputAxis >= 0) {
      if (input->shape[inputAxis] != 1)
        offset += coordinate * stride;
      stride *= input->shape[inputAxis];
    }
  }
  return offset;
}
} // namespace

void add_float(Tensor *input1, Tensor *input2, Tensor *output) {
  const auto *lhs = static_cast<const float *>(input1->location);
  const auto *rhs = static_cast<const float *>(input2->location);
  auto *result = static_cast<float *>(output->location);
  size_t count = 1;
  for (int axis = 0; axis < output->rank; ++axis)
    count *= output->shape[axis];
  for (size_t i = 0; i < count; ++i)
    result[i] = lhs[broadcastOffset(i, input1, output)] +
                rhs[broadcastOffset(i, input2, output)];
}
#endif
