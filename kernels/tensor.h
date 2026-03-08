#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <stddef.h> 

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t* shape; 
    void* location; 
    int rank; 
} Tensor;

#ifdef __cplusplus
}
#endif

#endif
