/**
 * Copyright 2022-2025, XGBoost Contributors
 */
#include "adaptive.h"

#include <algorithm>  // std::transform,std::find_if,std::copy,std::unique
#include <cmath>      // std::isnan
#include <cstddef>    // std::size_t
#include <iterator>   // std::distance
#include <vector>     // std::vector

#include "../common/algorithm.h"           // ArgSort
#include "../common/numeric.h"             // RunLengthEncode
#include "../common/stats.h"               // Quantile,WeightedQuantile
#include "../common/threading_utils.h"     // ParallelFor
#include "../common/transform_iterator.h"  // MakeIndexTransformIter
#include "../tree/sample_position.h"       // for SamplePosition
#include "xgboost/base.h"                  // bst_node_t
#include "xgboost/context.h"               // Context
#include "xgboost/data.h"                  // MetaInfo
#include "xgboost/host_device_vector.h"    // HostDeviceVector
#include "xgboost/linalg.h"                // MakeTensorView
#include "xgboost/span.h"                  // Span
#include "xgboost/tree_model.h"            // RegTree
#include "../tree/tree_view.h"             // for WalkTree

#if !defined(XGBOOST_USE_CUDA)
#include "../common/common.h"  // AssertGPUSupport
#endif                         // !defined(XGBOOST_USE_CUDA)

namespace xgboost::obj::detail {
void EncodeTreeLeafHost(Context const* ctx, RegTree const& tree,
                        std::vector<bst_node_t> const& position, std::vector<size_t>* p_nptr,
                        std::vector<bst_node_t>* p_nidx, std::vector<size_t>* p_ridx) {
  auto& nptr = *p_nptr;
  auto& nidx = *p_nidx;
  auto& ridx = *p_ridx;
  ridx = common::ArgSort<size_t>(ctx, position.cbegin(), position.cend());
  std::vector<bst_node_t> sorted_pos(position);
  // permutation
  for (size_t i = 0; i < position.size(); ++i) {
    sorted_pos[i] = position[ridx[i]];
  }
  // find the first non-sampled row
  size_t begin_pos = std::distance(
      sorted_pos.cbegin(),
      std::find_if(sorted_pos.cbegin(), sorted_pos.cend(),
                   [](bst_node_t nidx) { return tree::SamplePosition::IsValid(nidx); }));
  CHECK_LE(begin_pos, sorted_pos.size());

  std::vector<bst_node_t> leaf;
  tree::WalkTree(tree, [&](auto const& tree_view, bst_node_t nidx) {
    if (tree_view.IsLeaf(nidx)) {
      leaf.push_back(nidx);
    }
    return true;
  });

  if (begin_pos == sorted_pos.size()) {
    nidx = leaf;
    return;
  }

  auto beg_it = sorted_pos.begin() + begin_pos;
  common::RunLengthEncode(beg_it, sorted_pos.end(), &nptr);
  CHECK_GT(nptr.size(), 0);
  // skip the sampled rows in indptr
  std::transform(nptr.begin(), nptr.end(), nptr.begin(),
                 [begin_pos](size_t ptr) { return ptr + begin_pos; });

  size_t n_leaf = nptr.size() - 1;
  auto n_unique = std::unique(beg_it, sorted_pos.end()) - beg_it;
  CHECK_EQ(n_unique, n_leaf);
  nidx.resize(n_leaf);
  std::copy(beg_it, beg_it + n_unique, nidx.begin());

  if (n_leaf != leaf.size()) {
    FillMissingLeaf(leaf, &nidx, &nptr);
  }
}

void UpdateTreeLeafHostMultiTarget(Context const* ctx, std::vector<bst_node_t> const& position,
                                   std::int32_t group_idx, MetaInfo const& info,
                                   float learning_rate, HostDeviceVector<float> const& predt,
                                   std::vector<float> const& alphas, RegTree* p_tree) {
  auto& tree = *p_tree;
  CHECK(tree.IsMultiTarget());
  CHECK_EQ(static_cast<size_t>(group_idx), 0u);

  std::vector<bst_node_t> nidx;
  std::vector<size_t> nptr;
  std::vector<size_t> ridx;
  EncodeTreeLeafHost(ctx, tree, position, &nptr, &nidx, &ridx);
  size_t const n_leaf = nidx.size();
  bst_target_t const n_targets = tree.NumTargets();
  CHECK_EQ(alphas.size(), static_cast<size_t>(n_targets));

  if (nptr.empty()) {
    CHECK(false) << "Multi-target UpdateTreeLeaf: empty nptr (all rows sampled out) is not supported on CPU.";
  }

  auto const n_targets_sz = static_cast<size_t>(n_targets);
  std::vector<float> quantiles(n_leaf * n_targets_sz, 0.0f);
  auto const& h_node_idx = nidx;
  auto const& h_node_ptr = nptr;
  CHECK_LE(h_node_ptr.back(), info.num_row_);
  auto h_predt = linalg::MakeTensorView(ctx, predt.ConstHostSpan(), info.num_row_,
                                        predt.Size() / info.num_row_);
  auto h_labels_full = info.labels.HostView();

  std::int32_t n_threads;
  if constexpr (kHasParallelStableSort) {
    CHECK_GE(h_node_ptr.size(), 1);
    auto it = common::MakeIndexTransformIter(
        [&](std::size_t i) { return h_node_ptr[i + 1] - h_node_ptr[i]; });
    n_threads = std::any_of(it, it + h_node_ptr.size() - 1,
                            [](auto n) {
                              constexpr std::size_t kNeedParallelSort = 1ul << 19;
                              return n > kNeedParallelSort;
                            })
                    ? 1
                    : ctx->Threads();
  } else {
    n_threads = ctx->Threads();
  }
  collective::ApplyWithLabels(
      ctx, info, static_cast<void*>(quantiles.data()), quantiles.size() * sizeof(float), [&] {
        common::ParallelFor(n_leaf * n_targets_sz, n_threads, [&](size_t idx) {
          size_t const k = idx / n_targets_sz;
          size_t const t = idx % n_targets_sz;
          auto const nid = h_node_idx[k];
          CHECK(tree.GetMultiTargetTree()->IsLeaf(nid));
          CHECK_LT(k + 1, h_node_ptr.size());
          size_t const n = h_node_ptr[k + 1] - h_node_ptr[k];
          auto h_row_set = common::Span<size_t const>{ridx}.subspan(h_node_ptr[k], n);
          auto h_weights = linalg::MakeVec(&info.weights_);

          auto iter = common::MakeIndexTransformIter([&](size_t i) -> float {
            auto row_idx = h_row_set[i];
            auto y_col = std::min(static_cast<std::size_t>(t), info.labels.Shape(1) - 1);
            return h_labels_full(row_idx, y_col) -
                   h_predt(row_idx, static_cast<bst_target_t>(t));
          });
          auto w_it = common::MakeIndexTransformIter([&](size_t i) -> float {
            auto row_idx = h_row_set[i];
            return h_weights(row_idx);
          });

          float q{0};
          if (info.weights_.Empty()) {
            q = common::Quantile(ctx, alphas[t], iter, iter + n);
          } else {
            q = common::WeightedQuantile(ctx, alphas[t], iter, iter + n, w_it);
          }
          if (std::isnan(q)) {
            CHECK_EQ(n, 0u);
          }
          quantiles[idx] = q;
        });
      });

  for (float& q : quantiles) {
    q *= learning_rate;
  }
  tree.SetLeaves(std::move(nidx), common::Span<float const>{quantiles.data(), quantiles.size()});
}

void UpdateTreeLeafHost(Context const* ctx, std::vector<bst_node_t> const& position,
                        std::int32_t group_idx, MetaInfo const& info, float learning_rate,
                        HostDeviceVector<float> const& predt, float alpha, RegTree* p_tree) {
  auto& tree = *p_tree;
  CHECK(!tree.IsMultiTarget());

  std::vector<bst_node_t> nidx;
  std::vector<size_t> nptr;
  std::vector<size_t> ridx;
  EncodeTreeLeafHost(ctx, *p_tree, position, &nptr, &nidx, &ridx);
  size_t n_leaf = nidx.size();
  if (nptr.empty()) {
    std::vector<float> quantiles;
    UpdateLeafValues(ctx, &quantiles, nidx, info, learning_rate, p_tree);
    return;
  }

  CHECK(!position.empty());
  std::vector<float> quantiles(n_leaf, 0);
  std::vector<int32_t> n_valids(n_leaf, 0);

  auto const& h_node_idx = nidx;
  auto const& h_node_ptr = nptr;
  CHECK_LE(h_node_ptr.back(), info.num_row_);
  auto h_predt = linalg::MakeTensorView(ctx, predt.ConstHostSpan(), info.num_row_,
                                        predt.Size() / info.num_row_);

  // A heuristic to use parallel sort. If we use multiple threads here, the sorting is
  // performed using a single thread as openmp cannot allocate new threads inside a
  // parallel region.
  std::int32_t n_threads;
  if constexpr (kHasParallelStableSort) {
    CHECK_GE(h_node_ptr.size(), 1);
    auto it = common::MakeIndexTransformIter(
        [&](std::size_t i) { return h_node_ptr[i + 1] - h_node_ptr[i]; });
    n_threads = std::any_of(it, it + h_node_ptr.size() - 1,
                            [](auto n) {
                              constexpr std::size_t kNeedParallelSort = 1ul << 19;
                              return n > kNeedParallelSort;
                            })
                    ? 1
                    : ctx->Threads();
  } else {
    n_threads = ctx->Threads();
  }

  collective::ApplyWithLabels(
      ctx, info, static_cast<void*>(quantiles.data()), quantiles.size() * sizeof(float), [&] {
        // loop over each leaf
        common::ParallelFor(quantiles.size(), n_threads, [&](size_t k) {
          auto nidx = h_node_idx[k];
          CHECK(tree[nidx].IsLeaf());
          CHECK_LT(k + 1, h_node_ptr.size());
          size_t n = h_node_ptr[k + 1] - h_node_ptr[k];
          auto h_row_set = common::Span<size_t const>{ridx}.subspan(h_node_ptr[k], n);

          auto h_labels = info.labels.HostView().Slice(linalg::All(), IdxY(info, group_idx));
          auto h_weights = linalg::MakeVec(&info.weights_);

          auto iter = common::MakeIndexTransformIter([&](size_t i) -> float {
            auto row_idx = h_row_set[i];
            return h_labels(row_idx) - h_predt(row_idx, group_idx);
          });
          auto w_it = common::MakeIndexTransformIter([&](size_t i) -> float {
            auto row_idx = h_row_set[i];
            return h_weights(row_idx);
          });

          float q{0};
          if (info.weights_.Empty()) {
            q = common::Quantile(ctx, alpha, iter, iter + h_row_set.size());
          } else {
            q = common::WeightedQuantile(ctx, alpha, iter, iter + h_row_set.size(), w_it);
          }
          if (std::isnan(q)) {
            CHECK(h_row_set.empty());
          }
          quantiles.at(k) = q;
        });
      });

  UpdateLeafValues(ctx, &quantiles, nidx, info, learning_rate, p_tree);
}

#if !defined(XGBOOST_USE_CUDA) && !defined(XGBOOST_USE_HIP)
void UpdateTreeLeafDevice(Context const*, common::Span<bst_node_t const>, std::int32_t,
                          MetaInfo const&, float, HostDeviceVector<float> const&, float, RegTree*) {
  common::AssertGPUSupport();
}
#endif  // !defined(XGBOOST_USE_CUDA) && !defined(XGBOOST_USE_HIP)
}  // namespace xgboost::obj::detail

namespace xgboost::obj::cpu_impl {
void UpdateTreeLeaf(Context const* ctx, std::vector<bst_node_t> const& position,
                    bst_target_t group_idx, MetaInfo const& info, float learning_rate,
                    HostDeviceVector<float> const& predt, std::vector<float> const& alphas,
                    RegTree* p_tree) {
  if (p_tree->IsMultiTarget()) {
    detail::UpdateTreeLeafHostMultiTarget(ctx, position, static_cast<std::int32_t>(group_idx), info,
                                          learning_rate, predt, alphas, p_tree);
    return;
  }
  for (float alpha : alphas) {
    detail::UpdateTreeLeafHost(ctx, position, static_cast<std::int32_t>(group_idx), info,
                               learning_rate, predt, alpha, p_tree);
  }
}
}  // namespace xgboost::obj::cpu_impl
