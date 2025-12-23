/**
 * Copyright 2022-2023, XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_CUDA_CONTEXT_CUH_
#define XGBOOST_COMMON_CUDA_CONTEXT_CUH_

#if defined(XGBOOST_USE_CUDA)
#include <thrust/execution_policy.h>
#elif defined(XGBOOST_USE_HIP)
#include <thrust/system/hip/execution_policy.h>
#endif

#include "device_helpers.cuh"

namespace xgboost {
struct CUDAContext {
 private:
  dh::XGBCachingDeviceAllocator<char> caching_alloc_;
  dh::XGBDeviceAllocator<char> alloc_;

 public:
  /**
   * \brief Caching thrust policy.
   */
  auto CTP() const {
#if defined(XGBOOST_USE_CUDA)
#if THRUST_MAJOR_VERSION >= 2 || defined(XGBOOST_USE_RMM)
    return thrust::cuda::par_nosync(caching_alloc_).on(dh::DefaultStream());
#else
    return thrust::cuda::par(caching_alloc_).on(dh::DefaultStream());
#endif  // THRUST_MAJOR_VERSION >= 2
#elif defined(XGBOOST_USE_HIP)
	return thrust::hip::par_nosync(caching_alloc_).on(dh::DefaultStream());
#endif
  }
  /**
   * \brief Thrust policy without caching allocator.
   */
  auto TP() const {
#if defined(XGBOOST_USE_CUDA)
#if THRUST_MAJOR_VERSION >= 2
    return thrust::cuda::par_nosync(alloc_).on(dh::DefaultStream());
#else
    return thrust::cuda::par(alloc_).on(dh::DefaultStream());
#endif  // THRUST_MAJOR_VERSION >= 2
#elif defined(XGBOOST_USE_HIP)
	return thrust::hip::par_nosync(alloc_).on(dh::DefaultStream());
#endif
  }
  auto Stream() const { return dh::DefaultStream(); }
};
}  // namespace xgboost
#endif  // XGBOOST_COMMON_CUDA_CONTEXT_CUH_
