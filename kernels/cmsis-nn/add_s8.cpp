#include "kernels.h"
#include "cmsis-nn.h"
#ifdef ADD_S8
extern "C"{
void add_s8(Tensor* input1, Tensor* input2,
            int32_t* rescale_lhs_mult, int32_t* rescale_lhs_shift, Tensor* rescale_lhs,
            int32_t* rescale_temp_mult, int32_t* rescale_temp_shift, Tensor* rescale_temp,
            int32_t* rescale_rhs_mult, int32_t* rescale_rhs_shift, Tensor* rescale_rhs,
            int32_t* rescaleOp4_mult, int32_t* rescaleOp4_shift, Tensor* rescaleOp4,
            Tensor* addOp,int check_temp, int lhsInZp, int tempInZp, int rhsInZp, int outOutZp){

    int8_t* in1 = reinterpret_cast<int8_t*>(input1->location);
    int8_t* in2 = reinterpret_cast<int8_t*>(input2->location);
    int8_t* out = reinterpret_cast<int8_t*>(rescaleOp4->location);

    int32_t block_size = 1;
    for (int i = 0; i < 4; ++i) {
        block_size *= input1->shape[i];
    }

    int shift_count = rescaleOp4->shape[3];
    const int32_t input_1_offset = -lhsInZp;
    int32_t input_1_mult;
    int32_t input_1_shift[shift_count];

    const int32_t input_2_offset = -rhsInZp;
    int32_t input_2_mult;
    int32_t input_2_shift[shift_count];

    if(check_temp == 0){
        input_1_mult   = *rescale_temp_mult;
        input_2_mult   = *rescale_rhs_mult;
        for (int i = 0; i < shift_count; i++) {
            input_1_shift[i] = ((rescale_rhs_shift[i] - 30) * (-1));
            input_2_shift[i] = ((rescale_rhs_shift[i] - 31) * (-1));
        }
    }else{
        input_1_mult   = *rescale_lhs_mult;
        input_2_mult   = *rescale_temp_mult;
        for (int i = 0; i < shift_count; i++) {
            input_1_shift[i] = ((rescale_lhs_shift[i] - 31) * (-1));
            input_2_shift[i] = ((rescale_lhs_shift[i] - 30) * (-1));
        }
    }

    const int32_t out_offset = outOutZp;
    const int32_t out_mult = *rescaleOp4_mult;
    int32_t out_shift[shift_count];

    for (int i = 0; i < shift_count; i++) {
        out_shift[i] = ((rescaleOp4_shift[i] - 31) * (-1));
    }

    const int32_t left_shift = 0;

    const int32_t out_activation_min = -128;
    const int32_t out_activation_max = 127;

    arm_cmsis_nn_status status = arm_elementwise_add_s8(
        in1, in2,
        input_1_offset, input_1_mult, input_1_shift[0],
        input_2_offset, input_2_mult, input_2_shift[0],
        left_shift,
        out,
        out_offset, out_mult, out_shift[0],
        out_activation_min, out_activation_max,
        block_size
    );

}
}
#endif
