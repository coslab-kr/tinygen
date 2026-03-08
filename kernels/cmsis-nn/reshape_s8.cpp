#include "kernels.h"
#include "cmsis-nn.h"
#ifdef RESHAPE_S8
extern "C"{
void reshape_s8(Tensor* input, Tensor* output, int32_t* new_shape) {

    int32_t total_size = 1;
    for (int i = 0; i < output->rank; ++i) {
        total_size *= output->shape[i];
    }
    arm_reshape_s8(
        static_cast<int8_t*>(input->location),
        static_cast<int8_t*>(output->location),
        total_size
    );

}
}
#endif

