#include "kernels.h"
#include "cmsis-nn.h"
#ifdef TRANSPOSE_CONV2D_S1_S2_S8
extern "C"{
void  transpose_conv2d_s1_s2_s8(Tensor* input, Tensor* weight, Tensor* bias,
                      int32_t* rescale_multiplier, int32_t* rescale_shift,Tensor* rescale_output,
                      std::initializer_list<int> pad, std::initializer_list<int> stride,
                      std::initializer_list<int64_t> dilation,
                      int32_t inputZp, int32_t outputZp_rescale){


    cmsis_nn_transpose_conv_params tconv_params{};
    tconv_params.padding.w = *(pad.begin() + 1);
    tconv_params.padding.h = *(pad.begin() + 0);
    tconv_params.stride.w  = *(stride.begin() + 1);
    tconv_params.stride.h  = *(stride.begin() + 0);
    tconv_params.dilation.w = *(dilation.begin() + 1);
    tconv_params.dilation.h = *(dilation.begin() + 0);
    tconv_params.input_offset  = -inputZp;
    tconv_params.output_offset = outputZp_rescale;
    tconv_params.activation.min = -128;
    tconv_params.activation.max = 127;

    int mult_count = rescale_output->shape[3];
    int32_t shift[mult_count];

    for(int i = 0; i < mult_count; i++){
        shift[i] = 31 - rescale_shift[i];
    }

    cmsis_nn_per_channel_quant_params quant_params;
    quant_params.multiplier = rescale_multiplier;
    quant_params.shift      = shift.data();

    cmsis_nn_dims input_dims{input->shape[0], input->shape[1], input->shape[2], input->shape[3]};
    cmsis_nn_dims filter_dims{weight->shape[0], weight->shape[1], weight->shape[2], weight->shape[3]};
    cmsis_nn_dims bias_dims{1, 1, 1, bias->shape[0]};
    cmsis_nn_dims output_dims{rescale_output->shape[0], rescale_output->shape[1],
                              rescale_output->shape[2], rescale_output->shape[3]};

    int input_n = input_dims.n;
    int input_h = input_dims.h;
    int input_w = input_dims.w;
    int input_c = input_dims.c;

    int filter_h = filter_dims.h;
    int filter_w = filter_dims.w;
    int output_c = output_dims.c;

    int stride_h = tconv_params.stride.h;
    int stride_w = tconv_params.stride.w;
    int padding_h = tconv_params.padding.h;
    int padding_w = tconv_params.padding.w;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    void* buffer = nullptr;
    if (buf_size > 0) buffer = malloc(buf_size);
    cmsis_nn_context ctx{buffer, buf_size};

    size_t reverse_filter_size = filter_h * filter_w * input_c * output_c;
    int8_t* reversed_filter = (int8_t*)malloc(reverse_filter_size);
    if (!reversed_filter) {
        std::cerr << "[ERROR] malloc failed for reversed_filter\n";
        if (buffer) free(buffer);
        return;
    }

    const int8_t *in_ptr = static_cast<int8_t*>(weight->location);
    int8_t *out_ptr = reversed_filter;
    const int32_t filter_size = filter_h * filter_w * input_c;

    out_ptr += filter_size;
    for (int32_t i = 0; i < output_c; i++) {
        for (int32_t y = 0; y < filter_h; y++) {
            for (int32_t x = 0; x < filter_w; x++) {
                out_ptr -= input_c;
                memcpy(out_ptr, in_ptr, input_c * sizeof(int8_t));
                in_ptr += input_c;
            }
        }
        out_ptr += 2 * filter_size;
    }

    cmsis_nn_conv_params conv_params{};
    conv_params.padding.h = filter_h - 1 - padding_h;
    conv_params.padding.w = filter_w - 1 - padding_w;
    conv_params.input_offset  = tconv_params.input_offset;
    conv_params.output_offset = tconv_params.output_offset;
    conv_params.stride.h = 1;
    conv_params.stride.w = 1;
    conv_params.dilation.h = 1;
    conv_params.dilation.w = 1;
    conv_params.activation = tconv_params.activation;

    cmsis_nn_dims transposed_input_dims{input_n, input_h * stride_h, input_w * stride_w, input_c};
    cmsis_nn_dims upscale_dims{0, stride_h, stride_w, 0};

    arm_cmsis_nn_status status = arm_convolve_s8(
        &ctx,
        &conv_params,
        &quant_params,
        &transposed_input_dims,
        static_cast<int8_t*>(input->location),
        &filter_dims,
        reversed_filter,
        &bias_dims,
        static_cast<int32_t*>(bias->location),
        &upscale_dims,
        &output_dims,
        static_cast<int8_t*>(rescale_output->location));
    
    if (status != ARM_CMSIS_NN_SUCCESS) {
    }

    if (buffer) free(buffer);
    if (reversed_filter) free(reversed_filter);


}
}
#endif


