/**
 * Copyright 2017-2023 XGBoost contributors
 */
#pragma once

#if defined(XGBOOST_USE_HIP)

#include <hip/hip_runtime_api.h>

#ifndef hipEventWaitDefault
#define hipEventWaitDefault                                 0
#endif

#define cudaSuccess                                         hipSuccess
#define cudaError                                           hipError_t
#define cudaError_t                                         hipError_t
#define cudaGetLastError                                    hipGetLastError
#define cudaPeekAtLastError                                 hipPeekAtLastError
#define cudaErrorInvalidValue                               hipErrorInvalidValue
#define cudaGetErrorString                                  hipGetErrorString
#define CUresult                                            hipError_t

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
#define cudaEventWaitDefault                                hipEventWaitDefault

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
#define cudaMallocManaged                                   hipMallocManaged

#define cudaMemcpy                                          hipMemcpy
#define cudaMemcpyAsync                                     hipMemcpyAsync
#define cudaMemcpyDefault                                   hipMemcpyDefault
#define cudaMemcpyHostToDevice                              hipMemcpyHostToDevice
#define cudaMemcpyHostToHost                                hipMemcpyHostToHost
#define cudaMemcpyDeviceToHost                              hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice                            hipMemcpyDeviceToDevice
#define cudaMemsetAsync                                     hipMemsetAsync
#define cudaMemset                                          hipMemset
#define CUmemAllocationGranularity_flags                    hipMemAllocationGranularity_flags
#define cudaMemPrefetchAsync                                hipMemPrefetchAsync
#define cudaHostRegister                                    hipHostRegister
#define cudaHostUnregister                                  hipHostUnregister
#define cudaHostRegisterDefault                             hipHostRegisterDefault

#define cudaMemAdvise                                       hipMemAdvise
#define cudaMemAdviseSetReadMostly                          hipMemAdviseSetReadMostly
#define cudaMemAdviseSetPreferredLocation                   hipMemAdviseSetPreferredLocation
#define cudaMemAdviseSetAccessedBy                          hipMemAdviseSetAccessedBy

#define cudaPointerAttributes                               hipPointerAttribute_t 
#define cudaPointerGetAttributes                            hipPointerGetAttributes

/* hipMemoryTypeUnregistered not supported */
#define cudaMemoryTypeUnregistered                          hipMemoryTypeUnified
#define cudaMemoryTypeUnified                               hipMemoryTypeUnified
#define cudaMemoryTypeHost                                  hipMemoryTypeHost

#define cudaMemGetInfo                                      hipMemGetInfo
#define CUmemLocation                                       hipMemLocation
#define CUmemLocationType                                   hipMemLocationType
#define CUmemGenericAllocationHandle                        hipMemGenericAllocationHandle_t
#define CUmemAllocationProp                                 hipMemAllocationProp
#define CUmemAccessDesc                                     hipMemAccessDesc
#define CU_MEM_ALLOC_GRANULARITY_RECOMMENDED                hipMemAllocationGranularityRecommended
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE                  hipMemAccessFlagsProtReadWrite
#define CU_MEM_LOCATION_TYPE_DEVICE                         hipMemLocationTypeDevice
#define CU_MEM_LOCATION_TYPE_HOST_NUMA                      hipMemLocationTypeNuma

/* hack */
#ifndef hipMemLocationTypeNuma
#define hipMemLocationTypeNuma                              hipMemLocationTypeDevice
#endif

#define cudaFuncSetAttribute                                hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize         hipFuncAttributeMaxDynamicSharedMemorySize

#define cudaDevAttrMultiProcessorCount                      hipDeviceAttributeMultiprocessorCount
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor       hipOccupancyMaxActiveBlocksPerMultiprocessor

#define CUdevice                                            hipDevice_t
#define CUdeviceptr                                         hipDeviceptr_t
#define CUdevice_attribute                                  hipDeviceAttribute_t

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
