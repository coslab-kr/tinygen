#ifdef REDUCE_SUM
void reduce_sum(Tensor* input, Tensor* output, int axis) {

    int32_t in_shape[4];
    int32_t out_shape[4];
    for (int i = 0; i < 4; ++i) {
        in_shape[i] = input->shape[i];
        out_shape[i] = output->shape[i];
    }

    float* in_f = reinterpret_cast<float*>(input->location);
    float* out_f = reinterpret_cast<float*>(output->location);
    fill(out_f, out_f + (out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3]), 0.0f);

    int stride = 1;
    for (int i = axis + 1; i < output->rank; ++i) stride *= in_shape[i];
    int outer = 1;
    for (int i = 0; i < axis; ++i) outer *= in_shape[i];
    int reduce_dim = in_shape[axis];

    for (int o = 0; o < outer; ++o) {
        for (int s = 0; s < stride; ++s) {
            float sum_v = 0.0f;
            for (int r = 0; r < reduce_dim; ++r) {
                int idx = o * reduce_dim * stride + r * stride + s;
                sum_v += in_f[idx];
            }
            out_f[o * stride + s] = sum_v;
        }
    }

}
#endif
