#include "kernels.h"
#include "cmsis-nn.h"

#ifdef CMSIS_NN_CONV2D_F1X1_S8
extern "C"{
void conv2d_f1x1_s8(Tensor* input, Tensor* weight, Tensor* bias, 
                    int32_t* rescale_multiplier, int32_t* rescale_shift, Tensor* rescale_output,
                    Tensor* clamp_output, std::initializer_list<int> pad,
                    std::initializer_list<int> stride,  std::initializer_list<int64_t> dilation,
                    int32_t inputZp, int32_t outputZp_rescale, int32_t min_clamp, int32_t max_clamp){


    int mult_count = rescale_output->shape[3];
    int32_t shift[mult_count];

    cmsis_nn_conv_params conv_params;
    conv_params.padding.w = *(pad.begin() + 2);
    conv_params.padding.h = *(pad.begin() + 0);
    conv_params.stride.w = *(stride.begin() + 1);
    conv_params.stride.h = *(stride.begin() + 0);
    conv_params.dilation.h = *(dilation.begin() + 1);
    conv_params.dilation.w = *(dilation.begin() + 0);

    conv_params.input_offset = -inputZp;
    conv_params.output_offset = outputZp_rescale;
    conv_params.activation.min = min_clamp;
    conv_params.activation.max = max_clamp;

    const int in_ch  = input->shape[3];
    const int out_ch = conv2d_output->shape[3];

    cmsis_nn_per_channel_quant_params quant_params;

    for(int i = 0; i < mult_count; i++){
        shift[i] = 31-rescale_shift[i];

    }
    quant_params.multiplier = rescale_multiplier;
    quant_params.shift = shift;

    cmsis_nn_dims input_dims;
    input_dims.n = input->shape[0];
    input_dims.h = input->shape[1];
    input_dims.w = input->shape[2];
    input_dims.c = input->shape[3];

    cmsis_nn_dims filter_dims;
    filter_dims.n = weight->shape[0];
    filter_dims.h = weight->shape[1];
    filter_dims.w = weight->shape[2];
    filter_dims.c = weight->shape[3];

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
    int32_t buf_size = arm_convolve_wrapper_s8_get_buffer_size(
            &conv_params,
            &input_dims,
            &filter_dims,
            &output_dims);

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
    ctx.size = (uint32_t)buf_size;

    arm_convolve_1x1_s8(
            &ctx,
            &conv_params,
            &quant_params,
            &input_dims,
            static_cast<int8_t*>(input->location),
            &filter_dims,
            static_cast<int8_t*>(weight->location),
            &bias_dims,
            static_cast<int32_t*>(bias->location),
            &output_dims,
            static_cast<int8_t*>(rescale_output->location)
            );
    if (scratch_raw) free(scratch_raw);

}
}
#endif
