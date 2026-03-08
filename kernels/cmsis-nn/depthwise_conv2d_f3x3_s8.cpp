#include "kernels.h"
#include "cmsis-nn.h"
#ifdef DEPTHWISE_CONV2D_F3X3_S8
extern "C"{
void depthwise_conv2d_f3x3_s8(Tensor* input, Tensor* weight, Tensor* bias, 
                          int32_t* rescale_multiplier, int32_t* rescale_shift,
                          Tensor* rescale_output,
                          std::initializer_list<int> pad, std::initializer_list<int> stride,
                          std::initializer_list<int64_t> dilation,
                          int32_t inputZp, int32_t outputZp_rescale) {
    
    int mult_count = rescale_output->shape[3];    
    int32_t shift[mult_count];

    for (int i = 0; i < mult_count; i++) {
        shift[i] = 31 - rescale_shift[i];
    }

    cmsis_nn_dw_conv_params dw_params;
    dw_params.padding.w = *(pad.begin() + 2);
    dw_params.padding.h = *(pad.begin() + 0);
    dw_params.stride.w = *(stride.begin() + 1);
    dw_params.stride.h = *(stride.begin() + 0);
    dw_params.dilation.w = *(dilation.begin() + 0);
    dw_params.dilation.h = *(dilation.begin() + 1);
    dw_params.input_offset = -inputZp;
    dw_params.output_offset = outputZp_rescale;
    dw_params.activation.min = -128;
    dw_params.activation.max = 127;
    dw_params.ch_mult = 1;

    cmsis_nn_per_channel_quant_params quant_params;
    quant_params.multiplier = rescale_multiplier;
    quant_params.shift = shift;

    cmsis_nn_dims input_dims;
    input_dims.n = input->shape[0];
    input_dims.h = input->shape[1];
    input_dims.w = input->shape[2];
    input_dims.c = input->shape[3];

    cmsis_nn_dims filter_dims;
    filter_dims.n = 1;
    filter_dims.h = weight->shape[0];
    filter_dims.w = weight->shape[1];
    filter_dims.c = weight->shape[2];

    cmsis_nn_dims bias_dims;
    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = bias->shape[0];

    cmsis_nn_dims output_dims;
    output_dims.n = rescale_output->shape[0];
    output_dims.h = rescale_output->shape[1];
    output_dims.w = rescale_output->shape[2];
    output_dims.c = rescale_output->shape[3];

    cmsis_nn_context ctx{};
    ctx = {nullptr, 0};
    int32_t buf_size = arm_depthwise_conv_wrapper_s8_get_buffer_size(
        &dw_params, &input_dims, &filter_dims, &output_dims);

    void* scratch_raw = nullptr;
    if (buf_size > 0) {
        scratch_raw = malloc((size_t)buf_size + 15);
        if (!scratch_raw) {
            return;
        }
        void* scratch_aligned = (void*)(((uintptr_t)scratch_raw + 15) & ~((uintptr_t)15));
        ctx.buf = scratch_aligned;
    } else {
        ctx.buf = nullptr;
    }
    ctx.size = (uint32_t)(buf_size + 15);

    arm_depthwise_conv_3x3_s8(
        &ctx,
        &dw_params,
        &quant_params,
        &input_dims,
        static_cast<int8_t*>(input->location),
        &filter_dims,
        static_cast<int8_t*>(weight->location),
        &bias_dims,
        static_cast<int32_t*>(bias->location),
        &output_dims,
        static_cast<int8_t*>(rescale_output->location));


    if (scratch_raw) free(scratch_raw);
}
}
#endif
