#ifdef RECIPROCAL
void reciprocal(Tensor* input, Tensor* output) {

    int block_size = 1;
    for (int i = 0; i < input->rank; ++i)
        block_size *= input->shape[i];

    float* in_f = reinterpret_cast<float*>(input->location);
    float* out_f = reinterpret_cast<float*>(output->location);

    for (int i = 0; i < block_size; ++i) {
        if (fabs(in_f[i]) < 1e-9f)
            out_f[i] = 0.0f;
        else
            out_f[i] = 1.0f / in_f[i];
    }

}
#endif
