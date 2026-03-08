#ifdef EXP
void tensor_exp(Tensor* input, Tensor* output) {

    int block_size = 1;
    int rank = input->rank;
    for (int i = 0; i < rank; ++i)
        block_size *= input->shape[i];

    float* in_f = reinterpret_cast<float*>(input->location);
    float* out_f = reinterpret_cast<float*>(output->location);

    for (int i = 0; i < block_size; ++i)
        out_f[i] = std::exp(in_f[i]);

}
#endif
