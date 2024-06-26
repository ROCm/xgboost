/**
 * Copyright 2022-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>
#include <xgboost/linalg.h>  // Tensor,Vector

#include "../../../src/common/stats.h"
#include "../../../src/common/transform_iterator.h"  // common::MakeIndexTransformIter
#include "../helpers.h"

namespace xgboost::common {
TEST(Stats, Quantile) {
  Context ctx;
  {
    linalg::Tensor<float, 1> arr({20.f, 0.f, 15.f, 50.f, 40.f, 0.f, 35.f}, {7}, DeviceOrd::CPU());
    std::vector<size_t> index{0, 2, 3, 4, 6};
    auto h_arr = arr.HostView();
    auto beg = MakeIndexTransformIter([&](size_t i) { return h_arr(index[i]); });
    auto end = beg + index.size();
    auto q = Quantile(&ctx, 0.40f, beg, end);
    ASSERT_EQ(q, 26.0);

    q = Quantile(&ctx, 0.20f, beg, end);
    ASSERT_EQ(q, 16.0);

    q = Quantile(&ctx, 0.10f, beg, end);
    ASSERT_EQ(q, 15.0);
  }

  {
    std::vector<float> vec{1., 2., 3., 4., 5.};
    auto beg = MakeIndexTransformIter([&](size_t i) { return vec[i]; });
    auto end = beg + vec.size();
    auto q = Quantile(&ctx, 0.5f, beg, end);
    ASSERT_EQ(q, 3.);
  }
}

TEST(Stats, WeightedQuantile) {
  Context ctx;
  linalg::Tensor<float, 1> arr({1.f, 2.f, 3.f, 4.f, 5.f}, {5}, DeviceOrd::CPU());
  linalg::Tensor<float, 1> weight({1.f, 1.f, 1.f, 1.f, 1.f}, {5}, DeviceOrd::CPU());

  auto h_arr = arr.HostView();
  auto h_weight = weight.HostView();

  auto beg = MakeIndexTransformIter([&](size_t i) { return h_arr(i); });
  auto end = beg + arr.Size();
  auto w = MakeIndexTransformIter([&](size_t i) { return h_weight(i); });

  auto q = WeightedQuantile(&ctx, 0.50f, beg, end, w);
  ASSERT_EQ(q, 3);

  q = WeightedQuantile(&ctx, 0.0, beg, end, w);
  ASSERT_EQ(q, 1);

  q = WeightedQuantile(&ctx, 1.0, beg, end, w);
  ASSERT_EQ(q, 5);
}

TEST(Stats, Median) {
  Context ctx;

  {
    linalg::Tensor<float, 2> values{{.0f, .0f, 1.f, 2.f}, {4}, DeviceOrd::CPU()};
    HostDeviceVector<float> weights;
    linalg::Tensor<float, 1> out;
    Median(&ctx, values, weights, &out);
    auto m = out(0);
    ASSERT_EQ(m, .5f);

#if defined(XGBOOST_USE_CUDA) || defined(XGBOOST_USE_HIP)
    ctx = ctx.MakeCUDA(0);
    ASSERT_FALSE(ctx.IsCPU());
    Median(&ctx, values, weights, &out);
    m = out(0);
    ASSERT_EQ(m, .5f);
#endif  // defined(XGBOOST_USE_CUDA)
  }

  {
    ctx = ctx.MakeCPU();
    // 4x2 matrix
    linalg::Tensor<float, 2> values{{0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 2.f, 2.f}, {4, 2}, ctx.Device()};
    HostDeviceVector<float> weights;
    linalg::Tensor<float, 1> out;
    Median(&ctx, values, weights, &out);
    ASSERT_EQ(out(0), .5f);
    ASSERT_EQ(out(1), .5f);

#if defined(XGBOOST_USE_CUDA) || defined(XGBOOST_USE_HIP)
    ctx = ctx.MakeCUDA(0);
    Median(&ctx, values, weights, &out);
    ASSERT_EQ(out(0), .5f);
    ASSERT_EQ(out(1), .5f);
#endif  // defined(XGBOOST_USE_CUDA)
  }
}

namespace {
void TestMean(Context const* ctx) {
  std::size_t n{128};
  linalg::Vector<float> data({n}, ctx->Device());
  auto h_v = data.HostView().Values();
  std::iota(h_v.begin(), h_v.end(), .0f);

  auto nf = static_cast<float>(n);
  float mean = nf * (nf - 1) / 2 / n;

  linalg::Vector<float> res{{1}, ctx->Device()};
  Mean(ctx, data, &res);
  auto h_res = res.HostView();
  ASSERT_EQ(h_res.Size(), 1);
  ASSERT_EQ(mean, h_res(0));
}
}  // anonymous namespace

TEST(Stats, Mean) {
  Context ctx;
  TestMean(&ctx);
}

#if defined(XGBOOST_USE_CUDA) || defined(XGBOOST_USE_HIP)
TEST(Stats, GPUMean) {
  auto ctx = MakeCUDACtx(0);
  TestMean(&ctx);
}
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::common
