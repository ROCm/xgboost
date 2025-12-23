/**
 * Copyright 2018-2024, XGBoost contributors
 */

#include "common.h"

#if defined(XGBOOST_USE_CUDA)
#include <cuda_runtime_api.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#elif defined(XGBOOST_USE_HIP)
#include "cuda_to_hip.h"
#include <thrust/system/hip/error.h>
#include <thrust/system_error.h>
#endif

namespace dh {
void ThrowOnCudaError(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::string f;
    if (file != nullptr) {
      f = file;
    }

#if defined(XGBOOST_USE_CUDA)
    LOG(FATAL) << thrust::system_error(code, thrust::cuda_category(),f + ": " + std::to_string(line)).what();
#elif defined(XGBOOST_USE_HIP)
    LOG(FATAL) << thrust::system_error(code, thrust::hip_category(),f + ": " + std::to_string(line)).what();
#endif
  }
}
}  // namespace dh
