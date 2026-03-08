#ifdef ADD_FLOAT
void add_float(Tensor* input1, Tensor* input2, Tensor* output){

    bool check = true;
    int32_t block_size = 1;
    int32_t input1_size = 1;
    int32_t input2_size=1;

    for (int i = 0; i < output->rank; ++i){
        block_size *= output->shape[i];
        input1_size *= input1->shape[i];
        input2_size *= input2->shape[i];
    }

    float* in1_f = reinterpret_cast<float*>(input1->location);
    float* in2_f = reinterpret_cast<float*>(input2->location);
    vector<float> bc_v(block_size);
    std::vector<float> out_q(block_size);

    if(input1_size>input2_size){
        check = true;
        if(input2_size ==1){
            fill(bc_v.begin(), bc_v.end(), in2_f[0]);
        }else{
            for (int i = 0; i < block_size; ++i) {
                bc_v[i] = static_cast<float>(roundf(in2_f[i]));
            }
        }
    }else{
        check = false;
        if(input1_size ==1){
            fill(bc_v.begin(), bc_v.end(), in1_f[0]);
        }else{
            for (int i = 0; i < block_size; ++i) {
                bc_v[i] = static_cast<float>(roundf(in1_f[i]));
            }
        }
    }
    if(check){
        for (int i = 0; i < block_size; ++i) {
            out_q[i] = in1_f[i] + bc_v[i];
        }
    }else{
        for (int i = 0; i < block_size; ++i) {
            out_q[i] = in2_f[i] + bc_v[i];
        }

    }
    float* out_f = reinterpret_cast<float*>(output->location);
    for (int i = 0; i < block_size; ++i) {
        out_f[i] = out_q[i];
    }
}
#endif
