/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#include "gpu/cambricon/cnnl_matmul.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

#include "gpu/cambricon/cnnl_matmul_executor.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/cambricon/sycl_bang_scoped_context.hpp"
#include "gpu/cambricon/sycl_bang_stream.hpp"
#include "gpu/cambricon/sycl_bang_stream_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

status_t cnnl_matmul_t::execute(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    const bool with_bias = matmul_impl_->with_bias();
    const bool has_runtime_args = matmul_impl_->has_runtime_params();

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
    const auto bias_d = with_bias
            ? ctx.memory_mdw(DNNL_ARG_BIAS, pd()->weights_md(1))
            : nullptr;

    status_t status;
    if (has_runtime_args) {
        // Initialise all runtime parameters
        status = matmul_impl_->init_parameters(src_d, weights_d, dst_d, bias_d);
        if (status != status::success) return status;
    }

    cambricon::sycl_bang_stream_t *bang_stream
            = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

    status = executor_->execute(
            ctx, ctx.stream()->engine(), matmul_impl_);

    if (has_runtime_args) {
        auto &evts = bang_stream->sycl_ctx().get_sycl_deps().events;
        for (auto e : evts) {
            e.wait();
        }

        matmul_impl_->cleanup();
    }

    return status;
}

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl
