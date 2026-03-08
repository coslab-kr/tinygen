#ifdef CAST
void cast(Tensor* input, Tensor* output, int quant_check) {
    int total_size = 1;
    for (int i = 0; i < input->rank; i++) {
        total_size *= input->shape[i];
    }

    // ---- i8 → f32 ----
    if (quant_check == 0) {
        int8_t* input_data = static_cast<int8_t*>(input->location);
        float* output_data = static_cast<float*>(output->location);

        for (int i = 0; i < total_size; i++) {
            output_data[i] = static_cast<float>(input_data[i]);
        }

    }

    // ---- f32 → i8 ----
    else if (quant_check == 1) {
        float* input_data = static_cast<float*>(input->location);
        int8_t* output_data = static_cast<int8_t*>(output->location);

        for (int i = 0; i < total_size; i++) {
            int32_t val = static_cast<int32_t>(std::round(input_data[i]));
            val = std::max(-128, std::min(127, val)); // clamp
            output_data[i] = static_cast<int8_t>(val);
        }

    }

    else {
        std::cerr << "[ERROR] Unsupported cast type: quant_check="
                  << quant_check << "\n";
    }
}
#endif
