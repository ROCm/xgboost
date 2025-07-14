/*!
 * Copyright 2019-2020 by Contributors
 * \file survival_metric.cc
 * \brief Metrics for survival analysis
 * \author Avinash Barnwal, Hyunsu Cho and Toby Hocking
 */

// Dummy file to keep the CUDA conditional compile trick.
#if !defined(XGBOOST_USE_CUDA) && !defined(XGBOOST_USE_HIP)
#include "survival_metric.cu"
#endif  // !defined(XGBOOST_USE_CUDA) && !defined(XGBOOST_USE_HIP)
