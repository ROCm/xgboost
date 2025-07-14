/*!
 * Copyright 2018 XGBoost contributors
 */

// Dummy file to keep the CUDA conditional compile trick.

#include <dmlc/registry.h>
namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(hinge_obj);

}  // namespace obj
}  // namespace xgboost

#if !defined(XGBOOST_USE_CUDA) && !defined(XGBOOST_USE_HIP)
#include "hinge.cu"
#endif  // XGBOOST_USE_CUDA && XGBOOST_USE_HIP
