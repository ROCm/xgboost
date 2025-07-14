/**
 * Copyright 2017-2023 XGBoost contributors
 */
#pragma once

#if defined(XGBOOST_USE_HIP)

#define cudaSuccess                                         hipSuccess
#define cudaError                                           hipError_t
#define cudaError_t                                         hipError_t
#define cudaGetLastError                                    hipGetLastError
#define cudaPeekAtLastError                                 hipPeekAtLastError
#define cudaErrorInvalidValue                               hipErrorInvalidValue

#define cudaStream_t                                        hipStream_t
#define cudaStreamCreate                                    hipStreamCreate
#define cudaStreamCreateWithFlags                           hipStreamCreateWithFlags
#define cudaStreamDestroy                                   hipStreamDestroy
#define cudaStreamWaitEvent                                 hipStreamWaitEvent
#define cudaStreamSynchronize                               hipStreamSynchronize

#define cudaStreamLegacy                                    hipStreamLegacy
#define cudaStreamPerThread                                 hipStreamPerThread
#define hipStreamLegacyWkRd                                 0

#define cudaEvent_t                                         hipEvent_t
#define cudaEventCreate                                     hipEventCreate
#define cudaEventCreateWithFlags                            hipEventCreateWithFlags
#define cudaEventDestroy                                    hipEventDestroy

#define cudaGetDevice                                       hipGetDevice
#define cudaSetDevice                                       hipSetDevice
#define cudaGetDeviceCount                                  hipGetDeviceCount
#define cudaDeviceSynchronize                               hipDeviceSynchronize

#define cudaGetDeviceProperties                             hipGetDeviceProperties
#define cudaDeviceGetAttribute                              hipDeviceGetAttribute

#define cudaMallocHost                                      hipMallocHost
#define cudaFreeHost                                        hipFreeHost
#define cudaMalloc                                          hipMalloc
#define cudaFree                                            hipFree

#define cudaMemcpy                                          hipMemcpy
#define cudaMemcpyAsync                                     hipMemcpyAsync
#define cudaMemcpyDefault                                   hipMemcpyDefault
#define cudaMemcpyHostToDevice                              hipMemcpyHostToDevice
#define cudaMemcpyHostToHost                                hipMemcpyHostToHost
#define cudaMemcpyDeviceToHost                              hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice                            hipMemcpyDeviceToDevice
#define cudaMemsetAsync                                     hipMemsetAsync
#define cudaMemset                                          hipMemset

#define cudaPointerAttributes                               hipPointerAttribute_t 
#define cudaPointerGetAttributes                            hipPointerGetAttributes

/* hipMemoryTypeUnregistered not supported */
#define cudaMemoryTypeUnregistered                          hipMemoryTypeUnified
#define cudaMemoryTypeUnified                               hipMemoryTypeUnified
#define cudaMemoryTypeHost                                  hipMemoryTypeHost

#define cudaMemGetInfo                                      hipMemGetInfo
#define cudaFuncSetAttribute                                hipFuncSetAttribute

#define cudaDevAttrMultiProcessorCount                      hipDeviceAttributeMultiprocessorCount
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor       hipOccupancyMaxActiveBlocksPerMultiprocessor

namespace thrust {
    namespace hip {
    }

    namespace cuda = thrust::hip;
}

namespace thrust {
#define cuda_category hip_category
}

namespace hipcub {
}

namespace cub = hipcub;

#endif
