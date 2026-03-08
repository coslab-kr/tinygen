#ifndef OPS_CNN_H
#define OPS_CNN_H

#include <initializer_list>
#include <cstdlib>
#include "tensor.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "arm_nn_types.h"

#ifndef __smlabb
#define __smlabb(a, b, c) ((int32_t)((a) * (b) + (c)))
#endif
#ifndef __smlatt
#define __smlatt(a, b, c) ((int32_t)((a) * (b) + (c)))
#endif
#ifndef __qadd
#define __qadd(a, b) ((a) + (b))
#endif

#ifdef __cplusplus
extern "C" {
#endif

void conv2d_f1x1_s1x1_s8( Tensor*, Tensor*, Tensor*, int32_t*, int32_t*, Tensor*, Tensor*,
                        std::initializer_list<int>, std::initializer_list<int>, 
                        std::initializer_list<int64_t>, int32_t, int32_t, int32_t, int32_t);

void conv2d_f1x1_s8(Tensor*, Tensor*, Tensor*, int32_t*, int32_t*, Tensor*, Tensor*,
                    std::initializer_list<int>, std::initializer_list<int>,
                    std::initializer_list<int64_t>, int32_t, int32_t, int32_t, int32_t);

void conv2d_f3x3_s8(Tensor*, Tensor*, Tensor*, int32_t*, int32_t*, Tensor*, Tensor*,
                std::initializer_list<int>, std::initializer_list<int>, std::initializer_list<int64_t>,
                int32_t, int32_t, int32_t, int32_t);

void conv2d_fnxn_s8(Tensor*, Tensor*, Tensor*, int32_t*, int32_t*, Tensor*, Tensor*,
                  std::initializer_list<int>, std::initializer_list<int>, std::initializer_list<int64_t>,
                  int32_t, int32_t, int32_t, int32_t);

void conv2d_fnxm_s8(Tensor*, Tensor*, Tensor*, int32_t*, int32_t*, Tensor*, Tensor*,
                std::initializer_list<int>, std::initializer_list<int>, std::initializer_list<int64_t>,
                int32_t, int32_t, int32_t, int32_t);

void depthwise_conv2d_f3x3_s8(Tensor*, Tensor*, Tensor*, int32_t*, int32_t*, Tensor*,
                          std::initializer_list<int>, std::initializer_list<int>, std::initializer_list<int64_t>,
                          int32_t, int32_t);

void transpose_conv2d_s1_s2_s8(Tensor*, Tensor*, Tensor*, int32_t*, int32_t*, Tensor*,
                        std::initializer_list<int>, std::initializer_list<int>, std::initializer_list<int64_t>,
                        int32_t, int32_t);

void max_pool_2d(Tensor*, Tensor*, std::initializer_list<int>, std::initializer_list<int>, std::initializer_list<int>);
void avg_pool_2d(Tensor*, Tensor*, std::initializer_list<int>, std::initializer_list<int>, std::initializer_list<int>);

void add_s8(Tensor*, Tensor*, int32_t*, int32_t*, Tensor*, int32_t*, int32_t*, Tensor*,
           int32_t*, int32_t*, Tensor*, int32_t*, int32_t*, Tensor*, Tensor*, int, int, int, int, int);

void concat_axis3(Tensor*, Tensor*, Tensor*, int32_t);
void pad_s8(Tensor*, Tensor*, Tensor*, Tensor*);
void transpose(Tensor*, Tensor*, Tensor*);
void reshape_s8(Tensor*, Tensor*, int32_t*);

#ifdef __cplusplus
}
#endif

#endif // OPS_CNN_H

