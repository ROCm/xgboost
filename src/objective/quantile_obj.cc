/**
 * Copyright 2023 by XGBoost Contributors
 */

// Dummy file to enable the CUDA conditional compile trick.

#include <dmlc/registry.h>
namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(quantile_obj);

}  // namespace obj
}  // namespace xgboost

#if !defined(XGBOOST_USE_CUDA) && !defined(XGBOOST_USE_HIP)
#include "quantile_obj.cu"
#endif  // !defined(XBGOOST_USE_CUDA) && !defined(XGBOOST_USE_HIP)
