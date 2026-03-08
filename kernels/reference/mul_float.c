#ifdef MUL_FLOAT
void mul(Tensor* input1, Tensor* input2, Tensor* output, int32_t shift){

    float* in1_f = reinterpret_cast<float*>(input1->location);
    float* in2_f = reinterpret_cast<float*>(input2->location);
    float* out_f = reinterpret_cast<float*>(output->location);

    int rank_out = output->rank;
    int rank1 = input1->rank;
    int rank2 = input2->rank;

    int total_size = 1;
    for (int i = 0; i < rank_out; i++)
        total_size *= output->shape[i];

    vector<int32_t> shape1(rank_out, 1);
    vector<int32_t> shape2(rank_out, 1);

    for (int i = 0; i < rank1; i++)
        shape1[rank_out - rank1 + i] = input1->shape[i];
    for (int i = 0; i < rank2; i++)
        shape2[rank_out - rank2 + i] = input2->shape[i];

    vector<int> stride1(rank_out), stride2(rank_out), stride_out(rank_out);
    stride1[rank_out - 1] = stride2[rank_out - 1] = stride_out[rank_out - 1] = 1;

    for (int i = rank_out - 2; i >= 0; i--) {
        stride1[i] = stride1[i + 1] * shape1[i + 1];
        stride2[i] = stride2[i + 1] * shape2[i + 1];
        stride_out[i] = stride_out[i + 1] * output->shape[i + 1];
    }

    for (int idx = 0; idx < total_size; idx++) {
        int remain = idx;
        int offset1 = 0, offset2 = 0;

        for (int d = 0; d < rank_out; d++) {
            int coord = remain / stride_out[d];
            remain %= stride_out[d];

            offset1 += (shape1[d] == 1 ? 0 : coord) * stride1[d];
            offset2 += (shape2[d] == 1 ? 0 : coord) * stride2[d];
        }

        out_f[idx] = in1_f[offset1] * in2_f[offset2];
    }

}
#endif
