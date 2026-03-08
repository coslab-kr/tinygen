#ifdef REDUCE_MAX
void reduce_max(Tensor* input, Tensor* output, int axis) {
    
    int32_t* in_shape = input->shape;
    int32_t rank = input->rank;
    float* in_f = reinterpret_cast<float*>(input->location);
    float* out_f = reinterpret_cast<float*>(output->location);

    int out_size = 1;
    for (int i = 0; i < output->rank; ++i) out_size *= output->shape[i];
    fill(out_f, out_f + out_size, -FLT_MAX);

    // ---- Case 1: 2D Tensor (N,H)
    if (rank == 2) {
        int N = in_shape[0];
        int H = in_shape[1];

        if (axis == 0) {
            for (int h = 0; h < H; ++h) {
                float max_val = -FLT_MAX;
                for (int n = 0; n < N; ++n)
                    max_val = max(max_val, in_f[n * H + h]);
                out_f[h] = max_val;
            }
        } else if (axis == 1) {
            for (int n = 0; n < N; ++n) {
                float max_val = -FLT_MAX;
                for (int h = 0; h < H; ++h)
                    max_val = max(max_val, in_f[n * H + h]);
                out_f[n] = max_val;
            }
        } else {
            cout << "[ERROR] Unsupported axis for rank-2: " << axis << endl;
        }
    }

    // ---- Case 2: 4D Tensor (N,H,W,C)
    else if (rank == 4) {
        int N = in_shape[0];
        int H = in_shape[1];
        int W = in_shape[2];
        int C = in_shape[3];

        if (axis == 0) {
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                    for (int c = 0; c < C; c++) {
                        float max_val = -FLT_MAX;
                        for (int n = 0; n < N; n++) {
                            int idx = ((n * H + h) * W + w) * C + c;
                            max_val = max(max_val, in_f[idx]);
                        }
                        int out_idx = ((0 * H + h) * W + w) * C + c;
                        out_f[out_idx] = max_val;
                    }
        } else if (axis == 1) {
            for (int n = 0; n < N; n++)
                for (int w = 0; w < W; w++)
                    for (int c = 0; c < C; c++) {
                        float max_val = -FLT_MAX;
                        for (int h = 0; h < H; h++) {
                            int idx = ((n * H + h) * W + w) * C + c;
                            max_val = max(max_val, in_f[idx]);
                        }
                        int out_idx = ((n * output->shape[1] + 0) * W + w) * C + c;
                        out_f[out_idx] = max_val;
                    }
        } else if (axis == 2) {
            for (int n = 0; n < N; n++)
                for (int h = 0; h < H; h++)
                    for (int c = 0; c < C; c++) {
                        float max_val = -FLT_MAX;
                        for (int w = 0; w < W; w++) {
                            int idx = ((n * H + h) * W + w) * C + c;
                            max_val = max(max_val, in_f[idx]);
                        }
                        int out_idx = ((n * H + h) * output->shape[2] + 0) * C + c;
                        out_f[out_idx] = max_val;
                    }
        } else if (axis == 3) {
            for (int n = 0; n < N; n++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++) {
                        float max_val = -FLT_MAX;
                        for (int c = 0; c < C; c++) {
                            int idx = ((n * H + h) * W + w) * C + c;
                            max_val = max(max_val, in_f[idx]);
                        }
                        int out_idx = ((n * H + h) * W + w);
                        out_f[out_idx] = max_val;
                    }
        } else {
            cout << "[ERROR] Unsupported axis for rank-4: " << axis << endl;
        }
    }

    else {
        cout << "[ERROR] Unsupported tensor rank: " << rank << endl;
    }

}
#endif
