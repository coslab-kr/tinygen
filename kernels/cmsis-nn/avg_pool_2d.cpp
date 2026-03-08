#include "kernels.h"
#include "cmsis-nn.h"

#ifdef AVG_POOL_2D
extern "C"{
void avg_pool_2d(Tensor* input, Tensor* output, std::initializer_list<int> pad,
                 std::initializer_list<int> stride, std::initializer_list<int> kernel) {

    cmsis_nn_context ctx = {nullptr, 0};

    cmsis_nn_pool_params pool_params;
    pool_params.stride.w = *(stride.begin() + 1);
    pool_params.stride.h = *(stride.begin() + 0);
    pool_params.padding.w = *(pad.begin() + 1);
    pool_params.padding.h = *(pad.begin() + 0);
    pool_params.activation.min = -128;
    pool_params.activation.max = 127;

    cmsis_nn_dims input_dims;
    input_dims.n = input->shape[0];
    input_dims.h = input->shape[1];
    input_dims.w = input->shape[2];
    input_dims.c = input->shape[3];

    cmsis_nn_dims filter_dims;
    filter_dims.n = 1;
    filter_dims.h = *(kernel.begin() + 0);
    filter_dims.w = *(kernel.begin() + 1);
    filter_dims.c = 1;

    cmsis_nn_dims output_dims;
    output_dims.n = output->shape[0];
    output_dims.h = output->shape[1];
    output_dims.w = output->shape[2];
    output_dims.c = output->shape[3];

    arm_avgpool_s8(
        &ctx,
        &pool_params,
        &input_dims,
        static_cast<int8_t*>(input->location),
        &filter_dims,
        &output_dims,
        static_cast<int8_t*>(output->location)
    );

}
}
#endif

