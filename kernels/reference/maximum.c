#ifdef MAXIMUM
static inline int32_t requantize(int32_t value, int32_t multiplier, int32_t shift)
{
    int64_t prod = (int64_t)value * (int64_t)multiplier;
    int32_t result = (int32_t)((prod + (1ll << (shift - 1))) >> shift);
    return result;
}

void maximum(Tensor* input1,
            int32_t* rescaleOp1_mult, int32_t* rescaleOp1_shift, Tensor* rescaleOp1,
            int32_t* rescaleOp2_mult, int32_t* rescaleOp2_shift, Tensor* rescaleOp2,
            int32_t* rescaleOp3_mult, int32_t* rescaleOp3_shift, Tensor* rescaleOp3,
            Tensor* maxOp, int Op1_InZp, int Op2_InZp, int Op3_OutZp){

    int32_t total_size = 1;
    for (int i = 0; i < input1->rank; i++) {
        total_size *= input1->shape[i];
    }

    int8_t* in_data  = static_cast<int8_t*>(input1->location);
    int8_t* out_data = static_cast<int8_t*>(rescaleOp3->location);

    int32_t mult1 = rescaleOp1_mult[0];
    int32_t shift1 = rescaleOp1_shift[0];
    int32_t mult2 = rescaleOp2_mult[0];
    int32_t shift2 = rescaleOp2_shift[0];
    int32_t mult3 = rescaleOp3_mult[0];
    int32_t shift3 = rescaleOp3_shift[0];

    for (int i = 0; i < total_size; i++) {
        int32_t x = (int32_t)in_data[i];

        int32_t val1 = requantize(x - Op1_InZp, mult1, shift1);

        int32_t val2 = requantize(x - Op2_InZp, mult2, shift2);

        int32_t max_val = (val1 > val2) ? val1 : val2;

        int32_t val_out = requantize(max_val, mult3, shift3) + Op3_OutZp;

        if (val_out > 127) val_out = 127;
        if (val_out < -128) val_out = -128;

        out_data[i] = (int8_t)val_out;
    }
}
#endif
