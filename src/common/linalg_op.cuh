/**
 * Copyright 2021-2025, XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_LINALG_OP_CUH_
#define XGBOOST_COMMON_LINALG_OP_CUH_

#include <thrust/iterator/counting_iterator.h>  // for counting_iterator
#include <thrust/iterator/zip_iterator.h>       // for make_zip_iterator
#include <thrust/transform.h>                   // for transform

#include <cstdint>            // for int32_t
#include <cstdlib>            // for size_t
#if defined(XGBOOST_USE_HIP)
#include <iterator>
#include <tuple>
namespace cuda { namespace std { using namespace ::std; } }
#else
#include <cuda/std/iterator>  // for iterator_traits
#include <cuda/std/tuple>     // for get
#include <cuda/std/version>   // for CCCL_MINOR_VERSION
#endif
#include <tuple>              // for apply

#include "cuda_context.cuh"
#include "device_helpers.cuh"  // for LaunchN
#include "type.h"              // for GetValueT
#include "xgboost/context.h"   // for Context
#include "xgboost/linalg.h"
#include "linalg_op.h"  // for ElementWiseKernelHost    // for TensorView

#if defined(XGBOOST_USE_CUDA) && ((CCCL_MAJOR_VERSION >= 3) || (CCCL_MAJOR_VERSION >= 2 && CCCL_MINOR_VERSION >= 8))
#define xgboost_CCCL_HAS_PROCLAIM_COPYABLE 1
// CCCL 2.8.0 | CUDA 12.9
#include <cuda/functional>  // for proclaim_copyable_arguments
#endif

namespace xgboost::linalg {
namespace cuda_impl {
// Use template specialization to dispatch, Windows + CUDA 11.8 doesn't support extended
// lambda inside constexpr if
template <typename T, std::int32_t D>
struct ElementWiseImpl {
  template <typename Fn>
  void operator()(TensorView<T, D> t, Fn&& fn, cudaStream_t s) {
    static_assert(D > 1);
    dh::LaunchN(t.Size(), s, [=] __device__(std::size_t i) mutable {
      std::apply(fn, linalg::UnravelIndex(i, t.Shape()));
    });
  }
};

template <typename T>
struct ElementWiseImpl<T, 1> {
  template <typename Fn>
  void operator()(TensorView<T, 1> t, Fn&& fn, cudaStream_t s) {
    dh::LaunchN(t.Size(), s, [=] __device__(std::size_t i) mutable { fn(i); });
  }
};

template <typename T, std::int32_t D, typename Fn>
void ElementWiseKernel(TensorView<T, D> t, Fn&& fn, cudaStream_t s = nullptr) {
  dh::safe_cuda(cudaSetDevice(t.Device().ordinal));
  ElementWiseImpl<T, D>{}(t, fn, s);
}

template <typename T, std::int32_t D, typename Fn>
void TransformIdxKernel(Context const* ctx, TensorView<T, D> t, Fn&& fn) {
  TransformIdxKernel(ctx->CUDACtx(), t, std::forward<Fn>(fn));
}

template <typename T, std::int32_t D, typename Fn>
void TransformIdxKernel(CUDAContext const* ctx, TensorView<T, D> t, Fn&& fn) {
  dh::safe_cuda(cudaSetDevice(t.Device().ordinal));
  auto s = ctx->Stream();
  if (t.Contiguous()) {
    auto ptr = t.Values().data();
    auto it =
        thrust::make_zip_iterator(thrust::make_counting_iterator(static_cast<std::size_t>(0)), ptr);
    using Tuple = typename cuda::std::iterator_traits<common::GetValueT<decltype(it)>>::value_type;
    thrust::transform(ctx->CTP(), it, it + t.Size(), ptr,
                      [=] XGBOOST_DEVICE(Tuple const& tup) {
                        return fn(thrust::get<0>(tup), thrust::get<1>(tup));
                      });
  } else {
    dh::LaunchN(t.Size(), s, [=] __device__(size_t i) mutable {
      T& v = std::apply(t, UnravelIndex(i, t.Shape()));
      v = fn(i, v);
    });
  }
}

template <typename T, std::int32_t D, typename Fn>
void TransformKernel(Context const* ctx, TensorView<T, D> t, Fn&& fn) {
  dh::safe_cuda(cudaSetDevice(t.Device().ordinal));
  auto s = ctx->CUDACtx()->Stream();
  if (t.Contiguous()) {
    auto ptr = t.Values().data();
#if defined(xgboost_CCCL_HAS_PROCLAIM_COPYABLE)
    auto op = cuda::proclaim_copyable_arguments([=] XGBOOST_DEVICE(T const& v) { return fn(v); });
#else
    auto op = [=] XGBOOST_DEVICE(T const& v) {
      return fn(v);
    };
#endif
    thrust::transform(ctx->CUDACtx()->CTP(), ptr, ptr + t.Size(), ptr, op);
  } else {
    dh::LaunchN(t.Size(), s, [=] __device__(size_t i) mutable {
      T& v = std::apply(t, UnravelIndex(i, t.Shape()));
      v = fn(v);
    });
  }
}
}  // namespace cuda_impl

namespace detail {
template <typename T, std::int32_t D>
struct IterOp {
  TensorView<T, D> v;
  XGBOOST_DEVICE T& operator()(std::size_t i) { return std::apply(v, UnravelIndex(i, v.Shape())); }
};
}  // namespace detail

// naming: thrust begin
// returns a thrust iterator for a tensor view.
template <typename T, std::int32_t D>
auto tcbegin(TensorView<T, D> v) {  // NOLINT
  return thrust::make_transform_iterator(
      thrust::make_counting_iterator(0ul),
      detail::IterOp<std::add_const_t<std::remove_const_t<T>>, D>{v});
}

template <typename T, std::int32_t D>
auto tcend(TensorView<T, D> v) {  // NOLINT
  return tcbegin(v) + v.Size();
}

template <typename T, std::int32_t D>
auto tbegin(TensorView<T, D> v) {  // NOLINT
  return thrust::make_transform_iterator(thrust::make_counting_iterator(0ul),
                                         detail::IterOp<std::remove_const_t<T>, D>{v});
}

template <typename T, std::int32_t D>
auto tend(TensorView<T, D> v) {  // NOLINT
  return tbegin(v) + v.Size();
}
// Dispatcher for Context-based ElementWiseKernel (required by objectives)
template <typename T, std::int32_t D, typename Fn>
void ElementWiseKernel(Context const* ctx, TensorView<T, D> t, Fn&& fn) {
  if (ctx->IsCUDA()) {
    cuda_impl::ElementWiseKernel(t, fn, ctx->CUDACtx()->Stream());
  } else {
    ElementWiseKernelHost(t, ctx->Threads(), fn);
  }
}


// Dispatcher for Context-based TransformKernel (required by objectives, e.g. multiclass InitBaseScore)
template <typename T, std::int32_t D, typename Fn>
void TransformKernel(Context const* ctx, TensorView<T, D> t, Fn&& fn) {
  if (ctx->IsCUDA()) {
    cuda_impl::TransformKernel(ctx, t, std::forward<Fn>(fn));
  } else {
    ElementWiseKernelHost(t, ctx->Threads(), [&](std::size_t i) {
      T& v = std::apply(t, UnravelIndex(i, t.Shape()));
      v = fn(v);
    });
  }
}

inline void LogE(Context const* ctx, linalg::VectorView<float> x, float rt_eps = 0.0f) {
  CHECK_EQ(x.Device().ordinal, ctx->Device().ordinal);
  TransformKernel(ctx, x, [=] XGBOOST_DEVICE(float v) { return log(v + rt_eps); });
}

}  // namespace xgboost::linalg

#if defined(xgboost_CCCL_HAS_PROCLAIM_COPYABLE)
#undef xgboost_CCCL_HAS_PROCLAIM_COPYABLE
#endif  // defined(xgboost_CCCL_HAS_PROCLAIM_COPYABLE)

#endif  // XGBOOST_COMMON_LINALG_OP_CUH_
