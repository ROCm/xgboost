/**
 * Copyright 2025, XGBoost Contributors
 */
#if defined(XGBOOST_USE_CUDA)
#include <cuda/std/functional>  // for plus
#elif defined(XGBOOST_USE_HIP)
#include <functional>
#endif

#include <cstddef>              // for size_t

#include "cuda_context.cuh"
#include "device_helpers.cuh"
#include "optional_weight.h"
#include "xgboost/context.h"  // for Context

namespace xgboost::common::cuda_impl {
double SumOptionalWeights(Context const* ctx, OptionalWeights const& weights) {
  auto w_it = dh::MakeIndexTransformIter([=] XGBOOST_DEVICE(std::size_t i) { return weights[i]; });
#if defined(XGBOOST_USE_CUDA)
  return dh::Reduce(ctx->CUDACtx()->CTP(), w_it, w_it + weights.Size(), 0.0, cuda::std::plus{});
#elif defined(XGBOOST_USE_HIP)
  return dh::Reduce(ctx->CUDACtx()->CTP(), w_it, w_it + weights.Size(), 0.0, std::plus{});
#endif
}
}  // namespace xgboost::common::cuda_impl
