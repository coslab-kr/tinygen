#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

void model_init(); 
Tensor* model(); 

#ifdef __cplusplus
}
#endif
#endif

