import os
import shutil
import glob
import argparse

OP_LIST = "model_ops_list.txt"

ALWAYS_FILES = [
    "kernels/kernels.h",
    "kernels/model.h",
    "kernels/tensor.h",
]

ALWAYS_GLOBS = [
    "kernels/cmsis-nn/*.cpp",
    "kernels/cmsis-nn/*.h",
    "kernels/cmsis-nn/include/*.h",
    "kernels/reference/*.c",
    "kernels/reference/*.h",
]

cmsis_dic = {
    "conv2d_f3x3_s8": [
        "kernels/cmsis-nn/src/ConvolutionFunctions/arm_convolve_s8.c",
        "kernels/cmsis-nn/src/NNSupportFunctions/arm_q7_to_q15_with_offset.c",
        "kernels/cmsis-nn/src/NNSupportFunctions/arm_s8_to_s16_unordered_with_offset.c",
        "kernels/cmsis-nn/src/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c",
        "kernels/cmsis-nn/src/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.c",
        "kernels/cmsis-nn/src/ConvolutionFunctions/arm_nn_mat_mult_kernel_row_offset_s8_s16.c",
    ],
    "conv2d_f1x1_s8": [
        "kernels/cmsis-nn/src/ConvolutionFunctions/arm_convolve_1x1_s8.c",
        "kernels/cmsis-nn/src/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.c",
        "kernels/cmsis-nn/src/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c",
    ],
    "conv2d_f1x1_s1x1_s8": [
        "kernels/cmsis-nn/src/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.c",
        "kernels/cmsis-nn/src/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c",
        "kernels/cmsis-nn/src/ConvolutionFunctions/arm_convolve_1x1_s8_fast.c",
    ],
    "conv2d_fnxm_s8": [
        "kernels/cmsis-nn/src/ConvolutionFunctions/arm_convolve_s8.c",
        "kernels/cmsis-nn/src/NNSupportFunctions/arm_q7_to_q15_with_offset.c",
        "kernels/cmsis-nn/src/NNSupportFunctions/arm_s8_to_s16_unordered_with_offset.c",
        "kernels/cmsis-nn/src/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c",
        "kernels/cmsis-nn/src/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.c",
        "kernels/cmsis-nn/src/ConvolutionFunctions/arm_nn_mat_mult_kernel_row_offset_s8_s16.c",
    ],
    "avg_pool_2d": [
        "kernels/cmsis-nn/src/PoolingFunctions/arm_avgpool_s8.c",
        "kernels/cmsis-nn/src/PoolingFunctions/arm_avgpool_get_buffer_sizes_s8.c",
    ],
    "max_pool_2d": [
        "kernels/cmsis-nn/src/PoolingFunctions/arm_max_pool_s8.c",
    ],
    "pad_s8": [
        "kernels/cmsis-nn/src/PadFunctions/arm_pad_s8.c",
    ],
    "transpose_conv2d_s1_s2_s8": [
        "kernels/cmsis-nn/src/ConvolutionFunctions/arm_convolve_s8.c",
        "kernels/cmsis-nn/src/NNSupportFunctions/arm_q7_to_q15_with_offset.c",
        "kernels/cmsis-nn/src/NNSupportFunctions/arm_s8_to_s16_unordered_with_offset.c",
        "kernels/cmsis-nn/src/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c",
        "kernels/cmsis-nn/src/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.c",
    ],
    "depthwise_conv2d_f3x3_s8": [
        "kernels/cmsis-nn/src/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.c",
        "kernels/cmsis-nn/src/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s8.c",
    ],
    "reshape_s8": [
        "kernels/cmsis-nn/src/ReshapeFunctions/arm_reshape_s8.c",
    ],
    "add_s8": [
        "kernels/cmsis-nn/src/BasicMathFunctions/arm_elementwise_add_s8.c",
    ],
    "concat_axis3": [
        "kernels/cmsis-nn/src/ConcatenationFunctions/arm_concatenation_s8_z.c",
        "kernels/cmsis-nn/src/ConcatenationFunctions/arm_concatenation_s8_w.c",
        "kernels/cmsis-nn/src/ConcatenationFunctions/arm_concatenation_s8_y.c",
        "kernels/cmsis-nn/src/ConcatenationFunctions/arm_concatenation_s8_x.c",
    ],
}

SADD16_FALLBACK = r'''#ifndef __sadd16
#define __sadd16(a, b) ((int32_t)( \
    (int16_t)((a) & 0xFFFF) + (int16_t)((b) & 0xFFFF) | \
    (((int32_t)((int16_t)((a) >> 16) + (int16_t)((b) >> 16))) << 16)))
#endif

'''

def copy_flat(src, dst_dir, move=False):
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))

    if move:
        shutil.move(src, dst)
    else:
        shutil.copy(src, dst)

    if os.path.basename(src) == "arm_elementwise_add_s8.c":
        with open(dst, "r") as f:
            content = f.read()

        if "__sadd16" not in content:
            with open(dst, "w") as f:
                f.write(SADD16_FALLBACK)
                f.write(content)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True,
                        help="Model name, used to determine Arduino destination directory")
    args = parser.parse_args()

    MODEL_NAME = args.model
    DST_DIR = f"gen/{MODEL_NAME}"

    os.makedirs(DST_DIR, exist_ok=True)

    shutil.copytree(
        "kernels/cmsis-nn/include/Internal",
        os.path.join(DST_DIR, "Internal"),
        dirs_exist_ok=True
    )

    with open(OP_LIST) as f:
        ops = [line.split()[0] for line in f if line.strip()]


    for f in ALWAYS_FILES:
        if f.endswith("kernels.h"):
            copy_flat(f, DST_DIR, move=True)
        else:
            copy_flat(f, DST_DIR)

    for pattern in ALWAYS_GLOBS:
        for f in glob.glob(pattern):
            copy_flat(f, DST_DIR)

    for op in ops:
        if op not in cmsis_dic:
            continue

        for f in cmsis_dic[op]:
            if os.path.exists(f):
                copy_flat(f, DST_DIR)
            else:
                print(f"[WARN] Missing file: {f}")

if __name__ == "__main__":
    main()

