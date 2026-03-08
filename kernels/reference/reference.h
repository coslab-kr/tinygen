#ifndef REFERENCE_OPS_H
#define REFERENCE_OPS_H

#include <stdint.h>
#include "tensor.h"
#ifdef __cplusplus
extern "C" {
#endif

void add_float(Tensor*, Tensor*, Tensor*);
void sub_float(Tensor*, Tensor*, Tensor*);
void mul_float(Tensor*, Tensor*, Tensor*, int32_t);
void maximum(Tensor*, int32_t*, int32_t*, Tensor*, int32_t*, int32_t*, Tensor*,
               int32_t*, int32_t*, Tensor*, Tensor*, int, int, int);
void cast(Tensor*, Tensor*, int);
void reduce_max(Tensor*, Tensor*, int);
void reduce_sum(Tensor*, Tensor*, int);
void tensor_exp(Tensor*, Tensor*);
void reciprocal(Tensor*, Tensor*);

#ifdef __cplusplus
}
#endif

#endif  // REFERENCE_OPS_H

