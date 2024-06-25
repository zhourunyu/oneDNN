#===============================================================================
# Copyright 2020-2022 Intel Corporation
# Copyright 2020 Codeplay Software Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================
find_package(Threads REQUIRED)
find_package(BANG)

find_path(CNNL_INCLUDE_DIR  "cnnl.h"
        HINTS /usr/local/neuware/include)

find_library(CNNL_LIBRARY  cnnl
        HINTS /usr/local/neuware/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CNNL
    REQUIRED_VARS
        CNNL_INCLUDE_DIR
        CNNL_LIBRARY
)

if(CNNL_INCLUDE_DIR AND CNNL_LIBRARY)
    set(CNNL_FOUND TRUE)
    message(STATUS "CNNL_INCLUDE_DIR: ${CNNL_INCLUDE_DIR}")
    message(STATUS "CNNL_LIBRARY: ${CNNL_LIBRARY}")
    message(STATUS "find cnnl include & cnnl library")
else()
    message(FATAL_ERROR "cnnl not found!")
endif()


if(NOT TARGET cnnl::cnnl)
  add_library(cnnl::cnnl SHARED IMPORTED)
  set_target_properties(cnnl::cnnl PROPERTIES
      IMPORTED_LOCATION 
      ${CNNL_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES
      "${CNNL_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES
      "Threads::Threads;${CNNL_LIBRARY}"
      INTERFACE_COMPILE_DEFINITIONS
      CUDA_NO_HALF)
endif()