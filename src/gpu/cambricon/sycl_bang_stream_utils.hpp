/*******************************************************************************
* Copyright 2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_CAMBRICON_SYCL_BANG_STREAM_UTILS_HPP
#define GPU_CAMBRICON_SYCL_BANG_STREAM_UTILS_HPP

#include "gpu/cambricon/sycl_bang_scoped_context.hpp"
#include "gpu/cambricon/sycl_bang_stream.hpp"
#include "sycl/sycl_buffer_memory_storage.hpp"
#include "sycl/sycl_memory_storage_helper.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {
namespace stream_utils {

inline status_t copy_input_arg_to_host(const exec_ctx_t &ctx,
        sycl_bang_stream_t *stream, float *host_ptr, int arg, size_t size) {
    return stream->interop_task([&](::sycl::handler &cgh) {
        auto dev_arg = CTX_IN_SYCL_MEMORY(arg);

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine
                    = *utils::downcast<sycl_bang_engine_t *>(stream->engine());
            auto sc = bang_sycl_scoped_context_handler_t(sycl_engine);

            uint8_t *dev_ptr
                    = static_cast<uint8_t *>(dev_arg.get_native_pointer(ih));

            BANG_EXECUTE_FUNC(cnMemcpyAsync, (CNaddr)host_ptr,
                    (CNaddr)dev_ptr, size,
                    stream->get_underlying_stream());
            cnrtSyncDevice();
        });
    });
}

} // namespace stream_utils
} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_CAMBRICON_SYCL_BANG_STREAM_UTILS_HPP
