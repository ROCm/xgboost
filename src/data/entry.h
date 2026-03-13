/**
 *  Copyright 2019-2025, XGBoost Contributors
 */
#pragma once

#include "../common/math.h"  // for CheckNAN
#include "xgboost/base.h"    // for bst_idx_t
#include "xgboost/data.h"    // for Entry

namespace xgboost::data {
struct COOTuple {
  COOTuple() = default;
  XGBOOST_DEVICE COOTuple(
#if defined(XGBOOST_USE_HIP)
      ::xgboost::bst_idx_t row_idx, ::xgboost::bst_idx_t column_idx,
#else
      bst_idx_t row_idx, bst_idx_t column_idx,
#endif
      float value)
      : row_idx(row_idx), column_idx(column_idx), value(value) {}

#if defined(XGBOOST_USE_HIP)
  // HIP: qualify so lookup is correct when header is included from namespace enc
  ::xgboost::bst_idx_t row_idx{0};
  ::xgboost::bst_idx_t column_idx{0};
#else
  bst_idx_t row_idx{0};
  bst_idx_t column_idx{0};
#endif
  float value{0};
};

struct IsValidFunctor {
  float missing;

  XGBOOST_DEVICE explicit IsValidFunctor(float missing) : missing(missing) {}

  XGBOOST_DEVICE bool operator()(float value) const {
    return !(xgboost::common::CheckNAN(value) || value == missing);
  }

  XGBOOST_DEVICE bool operator()(const data::COOTuple& e) const {
    return !(xgboost::common::CheckNAN(e.value) || e.value == missing);
  }

  XGBOOST_DEVICE bool operator()(
#if defined(XGBOOST_USE_HIP)
      const ::xgboost::Entry& e
#else
      const Entry& e
#endif
  ) const {
    return !(xgboost::common::CheckNAN(e.fvalue) || e.fvalue == missing);
  }
};
}  // namespace xgboost::data
