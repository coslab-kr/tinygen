#include "kernels.h"
#include "cmsis-nn.h"
#ifdef TRANSPOSE
extern "C"{
void transpose(Tensor* input, Tensor* perms, Tensor* output){
    
    cmsis_nn_dims input_dims;
    input_dims.n = input->shape[0];
    input_dims.h = input->shape[1];
    input_dims.w = input->shape[2];
    input_dims.c = input->shape[3];

    cmsis_nn_dims output_dims;
    output_dims.n = output->shape[0];
    output_dims.h = output->shape[1];
    output_dims.w = output->shape[2];
    output_dims.c = output->shape[3];

    static uint32_t perm_data[4];
    for (int i = 0; i < 4; i++) {
        perm_data[i] = static_cast<int32_t*>(perms->location)[i];
    }

    cmsis_nn_transpose_params transpose_params = {
        perms->shape[0],
        perm_data
    };

    const int8_t* input_data = static_cast<const int8_t*>(input->location);
    int8_t* output_data = static_cast<int8_t*>(output->location);

    arm_cmsis_nn_status status = arm_transpose_s8(
        input_data,
        output_data,
        &input_dims,
        &output_dims,
        &transpose_params
    );

    if (status == ARM_CMSIS_NN_SUCCESS)
        std::cout << "Transpose completed successfully " << std::endl;
    else
        std::cout << "Transpose failed (status=" << status << ")" << std::endl;

}
}
#endif

