/**
 * Copyright 2017-2023 XGBoost contributors
 */
#pragma once

#if defined(XGBOOST_USE_HIP)

// Error codes and error handling
#define cudaSuccess                                         hipSuccess
#define cudaError                                           hipError_t
#define cudaError_t                                         hipError_t
#define cudaGetLastError                                    hipGetLastError
#define cudaPeekAtLastError                                 hipPeekAtLastError
#define cudaGetErrorString                                  hipGetErrorString
#define cudaErrorInvalidValue                               hipErrorInvalidValue

// Stream management
#define cudaStream_t                                        hipStream_t
#define cudaStreamCreate                                    hipStreamCreate
#define cudaStreamCreateWithFlags                           hipStreamCreateWithFlags
#define cudaStreamDestroy                                   hipStreamDestroy
#define cudaStreamWaitEvent                                 hipStreamWaitEvent
#define cudaStreamSynchronize                               hipStreamSynchronize
#define cudaStreamNonBlocking                               hipStreamNonBlocking
#define cudaStreamLegacy                                    hipStreamLegacy
#define cudaStreamPerThread                                 hipStreamPerThread
#define hipStreamLegacyWkRd                                 0

// Event management
#define cudaEvent_t                                         hipEvent_t
#define cudaEventCreate                                     hipEventCreate
#define cudaEventCreateWithFlags                            hipEventCreateWithFlags
#define cudaEventDestroy                                    hipEventDestroy
#define cudaEventSynchronize                                hipEventSynchronize
#define cudaEventDisableTiming                              hipEventDisableTiming
#define cudaEventDefault                                    hipEventDefault
#define cudaEventElapsedTime                                hipEventElapsedTime

// Device management
#define cudaGetDevice                                       hipGetDevice
#define cudaSetDevice                                       hipSetDevice
#define cudaGetDeviceCount                                  hipGetDeviceCount
#define cudaDeviceSynchronize                               hipDeviceSynchronize
#define cudaGetDeviceProperties                             hipGetDeviceProperties
#define cudaDeviceGetAttribute                              hipDeviceGetAttribute

// Device attributes
#define cudaDevAttrMultiProcessorCount                      hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrMaxSharedMemoryPerBlock                  hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaDevAttrMaxSharedMemoryPerBlockOptin             hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor       hipOccupancyMaxActiveBlocksPerMultiprocessor

// Memory allocation
#define cudaMalloc                                          hipMalloc
#define cudaMallocHost                                      hipMallocHost
#define cudaMallocManaged                                   hipMallocManaged
#define cudaMallocFromPoolAsync                             hipMallocFromPoolAsync
#define cudaFree                                            hipFree
#define cudaFreeHost                                        hipFreeHost
#define cudaFreeAsync                                       hipFreeAsync
#define cudaHostRegister                                    hipHostRegister
#define cudaHostUnregister                                  hipHostUnregister
#define cudaHostRegisterDefault                             hipHostRegisterDefault

// Memory operations
#define cudaMemcpy                                          hipMemcpy
#define cudaMemcpyAsync                                     hipMemcpyAsync
#define cudaMemcpyKind                                      hipMemcpyKind
#define cudaMemcpyDefault                                   hipMemcpyDefault
#define cudaMemcpyHostToDevice                              hipMemcpyHostToDevice
#define cudaMemcpyHostToHost                                hipMemcpyHostToHost
#define cudaMemcpyDeviceToHost                              hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice                            hipMemcpyDeviceToDevice
#define cudaMemset                                          hipMemset
#define cudaMemsetAsync                                     hipMemsetAsync
#define cudaMemGetInfo                                      hipMemGetInfo
#define cudaMemAdvise(...)                                  hipSuccess
#define cudaMemAdviseSetReadMostly                          0
#define cudaMemAdviseSetPreferredLocation                   0
#define cudaMemAdviseSetAccessedBy                          0
#define cudaMemPrefetchAsync                                hipMemPrefetchAsync

// Memory pool management
#define cudaMemPool_t                                       hipMemPool_t
#define cudaMemPoolProps                                    hipMemPoolProps
#define cudaMemPoolCreate                                   hipMemPoolCreate
#define cudaMemPoolSetAccess                                hipMemPoolSetAccess
#define cudaMemPoolDestroy                                  hipMemPoolDestroy

// Memory allocation types
#define cudaMemAllocationTypePinned                         hipMemAllocationTypePinned

// Memory location types
#define cudaMemLocationTypeHostNuma                         hipMemLocationTypeDevice
#define cudaMemLocationTypeDevice                           hipMemLocationTypeDevice

// Memory location struct - HIP doesn't have direct equivalent, provide dummy
struct cudaMemLocation {
  int type;
  int id;
};

// Memory handle types
#define cudaMemHandleTypeNone                               hipMemHandleTypeNone

// Memory access descriptors and flags
#define cudaMemAccessDesc                                   hipMemAccessDesc
#define cudaMemAccessFlagsProtReadWrite                     hipMemAccessFlagsProtReadWrite
#define CUmemAccessDesc                                     hipMemAccessDesc
#define CUmemAllocationProp                                 hipMemAllocationProp
#define CUmemLocationType                                   hipMemLocationType

// CUDA Driver API types (CU*) to HIP equivalents
#define CUmemGenericAllocationHandle                        hipMemGenericAllocationHandle_t
#define CUdeviceptr                                         hipDeviceptr_t
#define CUresult                                            hipError_t

// Pointer attributes
#define cudaPointerAttributes                               hipPointerAttribute_t
#define cudaPointerGetAttributes                            hipPointerGetAttributes

// Memory types
/* hipMemoryTypeUnregistered not supported */
#define cudaMemoryTypeUnregistered                          hipMemoryTypeUnregistered
#define cudaMemoryTypeUnified                               hipMemoryTypeUnified
#define cudaMemoryTypeHost                                  hipMemoryTypeHost

// Function attributes
#define cudaFuncSetAttribute                                hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize         hipFuncAttributeMaxDynamicSharedMemorySize

#define CUmemAllocationGranularity_flags 					hipMemAllocationGranularity_flags
#define CUdevice 											hipDevice_t
#define CUdevice_attribute 									hipDeviceAttribute_t
#define CUmemLocation 										hipMemLocation
#define CU_MEM_ALLOC_GRANULARITY_RECOMMENDED 				hipMemAllocationGranularityRecommended
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE	 				hipMemAccessFlagsProtReadWrite 
#define CU_MEM_LOCATION_TYPE_HOST_NUMA						hipMemLocationTypeDevice
#define CU_MEM_LOCATION_TYPE_DEVICE							hipMemLocationTypeDevice 

// Namespace aliases
namespace hipcub {
}

namespace cub = hipcub;

#endif

