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

#ifndef GPU_CAMBRICON_CNNL_MATMUL_EXECUTOR_HPP
#define GPU_CAMBRICON_CNNL_MATMUL_EXECUTOR_HPP

#include "gpu/cambricon/cnnl_matmul.hpp"
#include "gpu/cambricon/cnnl_matmul_impl.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/cambricon/sycl_bang_scoped_context.hpp"
#include "gpu/cambricon/sycl_bang_stream.hpp"
#include "sycl/sycl_memory_storage_helper.hpp"

#include <memory>

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct cnnl_matmul_exec_base_t {
    virtual status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<cnnl_matmul_impl_t> matmul_impl_)
            = 0;
    virtual ~cnnl_matmul_exec_base_t() = default;

protected:
    template <::sycl::access::mode bias_m>
    void interop_task(std::shared_ptr<cnnl_matmul_impl_t> matmul_impl_,
            engine_t *engine, ::sycl::handler &cgh,
            cambricon::sycl_bang_stream_t *bang_stream,
            impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read> arg_src,
            impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read>
                    arg_weights,
            impl::sycl::sycl_memory_arg_t<::sycl::access::mode::write> arg_dst,
            impl::sycl::sycl_memory_arg_t<bias_m> arg_bias,
            impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read>
                    arg_src_scale,
            impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read>
                    arg_wei_scale,
            impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read>
                    arg_dst_scale) {

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(
                    bang_stream->engine());
            auto sc = bang_sycl_scoped_context_handler_t(sycl_engine);
            auto native_stream = bang_stream->get_underlying_stream();
            auto cnnl_handle = bang_stream->get_cnnl_handle(native_stream);

            void *src = arg_src.get_native_pointer(ih);
            void *weights = arg_weights.get_native_pointer(ih);
            void *dst = arg_dst.get_native_pointer(ih);
            void *bias = arg_bias.get_native_pointer(ih);

            void *src_scale = arg_src_scale.get_native_pointer(ih);
            void *wei_scale = arg_wei_scale.get_native_pointer(ih);
            void *dst_scale = arg_dst_scale.get_native_pointer(ih);

            matmul_impl_->execute(cnnl_handle, src, weights,
                    dst, bias, src_scale, wei_scale, dst_scale);
        });
    }

    template <typename T, ::sycl::access::mode md, typename sc_t>
    void *maybe_cast_to_ptr(::sycl::accessor<T, 1, md> acc, sc_t &sc,
            const compat::interop_handle &ih) const {
        return sc.template memory<void *>(ih, acc);
    }

    template <typename sc_t>
    std::nullptr_t maybe_cast_to_ptr(std::nullptr_t acc, sc_t &,
            const compat::interop_handle &ih) const {
        return acc;
    }
};

struct cnnl_matmul_runtime_args_bias_exec_t : public cnnl_matmul_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<cnnl_matmul_impl_t> matmul_impl_) override {

        cambricon::sycl_bang_stream_t *bang_stream
                = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

        return bang_stream->interop_task([=](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_wt = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
            auto arg_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_BIAS);

            auto arg_src_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
            auto arg_wei_scale = CTX_IN_SYCL_MEMORY(
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
            auto arg_dst_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

            interop_task(matmul_impl_, engine, cgh, bang_stream, arg_src,
                    arg_wt, arg_dst, arg_bias, arg_src_scale, 
                    arg_wei_scale, arg_dst_scale);
        });
    }
};

struct cnnl_matmul_runtime_args_exec_t : public cnnl_matmul_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<cnnl_matmul_impl_t> matmul_impl_) override {

        cambricon::sycl_bang_stream_t *bang_stream
                = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

        return bang_stream->interop_task([=](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_wt = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);

            auto arg_bias = impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::read>();

            auto arg_src_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
            auto arg_wei_scale = CTX_IN_SYCL_MEMORY(
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
            auto arg_dst_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

            interop_task(matmul_impl_, engine, cgh, bang_stream, arg_src,
                    arg_wt, arg_dst, /*nullptr*/ arg_bias,
                    arg_src_scale, arg_wei_scale, arg_dst_scale);
        });
    }
};

struct cnnl_matmul_bias_exec_t : public cnnl_matmul_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<cnnl_matmul_impl_t> matmul_impl_) override {

        cambricon::sycl_bang_stream_t *bang_stream
                = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

        return bang_stream->interop_task([=](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_wt = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
            auto arg_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_BIAS);

            auto arg_src_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
            auto arg_wei_scale = CTX_IN_SYCL_MEMORY(
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
            auto arg_dst_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

            interop_task(matmul_impl_, engine, cgh, bang_stream, arg_src,
                    arg_wt, arg_dst, arg_bias, arg_src_scale, 
                    arg_wei_scale, arg_dst_scale);
        });
    }
};

struct cnnl_matmul_exec_t : public cnnl_matmul_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<cnnl_matmul_impl_t> matmul_impl_) override {

        cambricon::sycl_bang_stream_t *bang_stream
                = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

        return bang_stream->interop_task([=](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_wt = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);

            auto arg_bias = impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::read>();

            auto arg_src_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
            auto arg_wei_scale = CTX_IN_SYCL_MEMORY(
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
            auto arg_dst_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

            interop_task(matmul_impl_, engine, cgh, bang_stream, arg_src,
                    arg_wt, arg_dst, /*nullptr*/ arg_bias,
                    arg_src_scale, arg_wei_scale, arg_dst_scale);
        });
    }
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
