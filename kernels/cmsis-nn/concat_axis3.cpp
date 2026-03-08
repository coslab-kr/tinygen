#include "kernels.h"
#include "cmsis-nn.h"
#ifdef CONCAT_AXIS3
extern "C"{
void concat_axis3(Tensor* input1, Tensor* input2, Tensor* output, int32_t axis){

    int32_t N = input1->shape[0];
    int32_t H = input1->shape[1];
    int32_t W = input1->shape[2];
    int32_t C = input1->shape[3];
    int32_t N2 = input2->shape[0];
    int32_t H2 = input2->shape[1];
    int32_t W2 = input2->shape[2];
    int32_t C2 = input2->shape[3];

    int8_t* in1 = static_cast<int8_t*>(input1->location);
    int8_t* in2 = static_cast<int8_t*>(input2->location);
    int8_t* out = static_cast<int8_t*>(output->location);

        uint16_t input_x = H;
        uint16_t input_y = W;
        uint16_t input_z = C;
        uint16_t input_w = N;
        uint16_t output_z = C + C2;

        int offset = 0;

        for (int i = 0; i < input_x; i++) {
            for (int j = 0; j < input_y; j++) {

                const int8_t* in1_pixel = in1 + (i * input_y + j) * input_z;
                const int8_t* in2_pixel = in2 + (i * input_y + j) * C2;

                arm_concatenation_s8_z(in1_pixel,
                                   1, 1, input_z, input_w,
                                   out, output_z, offset);
                offset += input_z;

                arm_concatenation_s8_z(in2_pixel,
                                   1, 1, C2, input_w,
                                   out, output_z, offset);
                offset += C2;
            }
        }

}
}
#endif

