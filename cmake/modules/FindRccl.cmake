#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Tries to find RCCL headers and libraries.
#
# Usage of this module as follows:
#
#  find_package(RCCL)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  RCCL_ROOT - When set, this path is inspected instead of standard library
#              locations as the root of the RCCL installation.
#              The environment variable RCCL_ROOT overrides this variable.
#
# This module defines
#  Rccl_FOUND, whether rccl has been found
#  RCCL_INCLUDE_DIR, directory containing header
#  RCCL_LIBRARY, directory containing rccl library
#  RCCL_LIB_NAME, rccl library name
#  USE_RCCL_LIB_PATH, when set, RCCL_LIBRARY path is also inspected for the
#                     location of the rccl library. This would disable
#                     switching between static and shared.
#
# This module assumes that the user has already called find_package(HIP)

if(RCCL_LIBRARY)
  if(NOT USE_RCCL_LIB_PATH)
    # Don't cache RCCL_LIBRARY to enable switching between static and shared.
    unset(RCCL_LIBRARY CACHE)
  endif()
endif()

if(BUILD_WITH_SHARED_RCCL)
  # librccl.so
  set(RCCL_LIB_NAME rccl)
else()
  # librccl.a
  set(RCCL_LIB_NAME rccl)
endif()

find_path(RCCL_INCLUDE_DIR
  NAMES rccl/rccl.h
  HINTS ${RCCL_ROOT}/include $ENV{RCCL_ROOT}/include 
        /opt/rocm/include /opt/rocm/rccl/include
        $ENV{ROCM_PATH}/include $ENV{ROCM_PATH}/rccl/include)

if(USE_DLOPEN_RCCL)
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(Rccl DEFAULT_MSG RCCL_INCLUDE_DIR)

  mark_as_advanced(RCCL_INCLUDE_DIR)
else()
  find_library(RCCL_LIBRARY
    NAMES ${RCCL_LIB_NAME}
    HINTS ${RCCL_ROOT}/lib $ENV{RCCL_ROOT}/lib
          /opt/rocm/lib /opt/rocm/rccl/lib
          $ENV{ROCM_PATH}/lib $ENV{ROCM_PATH}/rccl/lib)

  message(STATUS "Using rccl library: ${RCCL_LIBRARY}")

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(Rccl DEFAULT_MSG
    RCCL_INCLUDE_DIR RCCL_LIBRARY)

  mark_as_advanced(
    RCCL_INCLUDE_DIR
    RCCL_LIBRARY
  )
endif()

# Set HIP_LIB_INSTALL_DIR if not already set
if(NOT HIP_LIB_INSTALL_DIR)
  if(DEFINED ENV{ROCM_PATH})
    set(HIP_LIB_INSTALL_DIR "$ENV{ROCM_PATH}/lib")
  else()
    set(HIP_LIB_INSTALL_DIR "/opt/rocm/lib")
  endif()
endif()
