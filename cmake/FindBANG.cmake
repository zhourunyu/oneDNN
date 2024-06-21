#===============================================================================
# Copyright 2022 Intel Corporation
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


find_path(
    BANG_INCLUDE_DIR "cnrt.h"
    HINTS /usr/local/neuware/include
)

find_library(
    BANG_LIBRARY cnrt
    HINTS /usr/local/neuware/lib64
)

if(BANG_INCLUDE_DIR AND BANG_LIBRARY)
    set(BANG_FOUND TRUE)
else()
    message(FATAL_ERROR "cnrt not found")
endif()
