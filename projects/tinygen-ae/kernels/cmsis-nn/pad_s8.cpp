#include "kernels.h"
#include "cmsis-nn.h"
#ifdef PAD_S8
extern "C"{
void pad_s8(Tensor* input, Tensor* padding, Tensor* pad_const, Tensor* output){

    const int8_t* input_data = (const int8_t*)(input->location);
    const int32_t* input_shape = input->shape;

    int8_t pad_value = *((int8_t*)(pad_const->location));

    const int* padding_vals = reinterpret_cast<const int*>(padding->location);

    cmsis_nn_dims input_size = {
        .n = input_shape[0],
        .h = input_shape[1],
        .w = input_shape[2],
        .c = input_shape[3]
    };

    cmsis_nn_dims pre_pad = {
        .n = padding_vals[0],
        .h = padding_vals[2],
        .w = padding_vals[4],
        .c = padding_vals[6]
    };

    cmsis_nn_dims post_pad = {
        .n = padding_vals[1],
        .h = padding_vals[3],
        .w = padding_vals[5],
        .c = padding_vals[7]
    };


    arm_cmsis_nn_status status = arm_pad_s8(
                input_data,
                static_cast<int8_t*>(output->location),
                pad_value,
                &input_size,
                &pre_pad,
                &post_pad
                );

}
}
#endif

