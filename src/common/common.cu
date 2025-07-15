/**
 * Copyright 2018-2024, XGBoost contributors
 */
#if defined(XGBOOST_USE_CUDA)
#include <thrust/system/cuda/error.h>
#elif defined(XGBOOST_USE_HIP)
#include <thrust/system/hip/error.h>
#endif
#include <thrust/system_error.h>

#include "common.h"
#include "cuda_to_hip.h"

namespace dh {
void ThrowOnCudaError(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::string f;
    if (file != nullptr) {
      f = file;
    }
    LOG(FATAL) << thrust::system_error(code, thrust::cuda_category(),
                                       f + ": " + std::to_string(line))
                      .what();
  }
}
}  // namespace dh
