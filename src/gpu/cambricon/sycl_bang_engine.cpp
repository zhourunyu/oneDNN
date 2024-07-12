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

#include "common/impl_list_item.hpp"
#include "common/utils.hpp"

#include "sycl/sycl_utils.hpp"


// TODO
// #include "gpu/cambricon/cnnl_batch_normalization.hpp"
// #include "gpu/cambricon/cnnl_binary.hpp"
// #include "gpu/cambricon/cnnl_conv_inner_product.hpp"
// #include "gpu/cambricon/cnnl_convolution.hpp"
// #include "gpu/cambricon/cnnl_deconvolution.hpp"
// #include "gpu/cambricon/cnnl_eltwise.hpp"
// #include "gpu/cambricon/cnnl_gemm_inner_product.hpp"
// #include "gpu/cambricon/cnnl_lrn.hpp"
// #include "gpu/cambricon/cnnl_matmul.hpp"
// #include "gpu/cambricon/cnnl_pooling.hpp"
// #include "gpu/cambricon/cnnl_reduction.hpp"
// #include "gpu/cambricon/cnnl_resampling.hpp"
#include "gpu/cambricon/cnnl_softmax.hpp"
#include "gpu/cambricon/sycl_bang_compat.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/cambricon/sycl_bang_scoped_context.hpp"
#include "gpu/cambricon/sycl_bang_stream.hpp"

#include "gpu/sycl/ref_batch_normalization.hpp"
#include "gpu/sycl/ref_binary.hpp"
#include "gpu/sycl/ref_eltwise.hpp"
#include "gpu/sycl/ref_layer_normalizations.hpp"
#include "gpu/sycl/ref_lrn.hpp"
#include "gpu/sycl/ref_pooling.hpp"
#include "gpu/sycl/ref_prelu.hpp"
#include "gpu/sycl/ref_resampling.hpp"
#include "gpu/sycl/ref_shuffle.hpp"
#include "gpu/sycl/ref_softmax.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

bool is_cambricon_mlu(const ::sycl::device &dev) {
    constexpr int cambricon_vendor_id = 0xcabc;
    return dev.is_gpu()
            && dev.get_info<::sycl::info::device::vendor_id>()
            == cambricon_vendor_id;
}

status_t bang_engine_create(engine_t **engine, engine_kind_t engine_kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index) {
    CHECK(cambricon::check_device(engine_kind));
    std::unique_ptr<cambricon::sycl_bang_engine_t, engine_deleter_t> bang_engine(
            (new cambricon::sycl_bang_engine_t(dev, ctx, index)));
    if (!bang_engine) return status::out_of_memory;

    CHECK(bang_engine->init());
    *engine = bang_engine.release();

    return status::success;
}

sycl_bang_engine_t::sycl_bang_engine_t(engine_kind_t kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index)
    : base_t(kind, dev, ctx, index) {
    underlying_context_type();
    set_cnnl_handle();
}

sycl_bang_engine_t::sycl_bang_engine_t(
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index)
    : sycl_bang_engine_t(engine_kind::gpu, dev, ctx, index) {
    assert(is_cambricon_mlu(dev));
}

status_t sycl_bang_engine_t::set_cnnl_handle() {
    // scoped context will make sure the top of the stack context is
    // the engine context while creating the cnnl handle.
    bang_sycl_scoped_context_handler_t sc(*this);
    cnnlHandle_t handle;
    CHECK(CNNL_EXECUTE_FUNC_S(cnnlCreate, &handle));
    cnnl_handle_.set(
            std::unique_ptr<cnnlHandle_t, void (*)(cnnlHandle_t *)>(
                    new cnnlHandle_t(handle), [](cnnlHandle_t *h) {
                        if (h != nullptr)
                            CNNL_EXECUTE_FUNC_V(cnnlDestroy, *h);
                        delete h;
                    }));
    handle = nullptr;
    return status::success;
}


CNcontext sycl_bang_engine_t::get_underlying_context() const {
    return compat::get_native<CNcontext>(context());
}

CNdev sycl_bang_engine_t::get_underlying_device() const {
    return compat::get_native<CNdev>(device());
}

status_t sycl_bang_engine_t::create_stream(stream_t **stream, unsigned flags) {
    return sycl_bang_stream_t::create_stream(stream, this, flags);
}

status_t sycl_bang_engine_t::create_stream(
        stream_t **stream, ::sycl::queue &queue) {
    return sycl_bang_stream_t::create_stream(stream, this, queue);
}

status_t sycl_bang_engine_t::underlying_context_type() {
    // this is a costly function which take avarage up to 75ms
    // on titanrx. So we must run it once and store the variable
    // in primary_context_;
    CNcontext primary;
    CNdev device_current;
    CNcontext desired = compat::get_native<CNcontext>(context());
    CNdev bang_device = compat::get_native<CNdev>(device());
        CNresult ret = cnCtxGetCurrent(&primary);
    if(ret != CN_SUCCESS){
        printf("%s@%d return %d FAILED\n",__func__, __LINE__,ret);
    }
    CNresult ret2 = cnCtxGetDevice(&device_current);
    if(ret2 != CN_SUCCESS){
        printf("%s@%d return %d FAILED\n",__func__, __LINE__,ret2);
    }
    primary_context_ = primary == desired && device_current == bang_device;
    return status::success;
}

cnnlHandle_t *sycl_bang_engine_t::get_cnnl_handle() {
    if (!cnnl_handle_.is_set()) set_cnnl_handle();
    return cnnl_handle_.get().get();
}


device_id_t sycl_bang_engine_t::device_id() const {
    return device_id_t(static_cast<int>(impl::sycl::backend_t::cambricon),
            static_cast<uint64_t>(compat::get_native<CNdev>(device())),
            static_cast<uint64_t>(0));
}

void sycl_bang_engine_t::activate_stream_cnnl(CNqueue bang_stream) {
    bang_sycl_scoped_context_handler_t sc(*this);
    cnrtQueue_t current_stream_id = nullptr;
    auto cnnl_handle = get_cnnl_handle();
    CNNL_EXECUTE_FUNC(cnnlGetQueue, *cnnl_handle, &current_stream_id);
    if (current_stream_id != bang_stream) {
        CNNL_EXECUTE_FUNC(cnnlSetQueue, *cnnl_handle, bang_stream);
    }
}


namespace {
using namespace dnnl::impl::data_type;

// clang-format off
constexpr dnnl::impl::impl_list_item_t sycl_bang_impl_list[] = {
        // Elementwise
        INSTANCE(sycl::ref_sycl_eltwise_fwd_t)
        INSTANCE(sycl::ref_sycl_eltwise_bwd_t)

        // Deconvolution

        // Convolution
        // INSTANCE(cnnl_convolution_fwd_t)
        // INSTANCE(cnnl_convolution_bwd_data_t)
        // INSTANCE(cnnl_convolution_bwd_weights_t)

        // Batch Normalization
        // INSTANCE(cnnl_batch_normalization_fwd_t)
        // INSTANCE(cnnl_batch_normalization_bwd_t)
        INSTANCE(sycl::ref_batch_normalization_fwd_t)
        INSTANCE(sycl::ref_batch_normalization_bwd_t)

        // Layer Normalization
        INSTANCE(sycl::ref_layer_normalization_fwd_t)
        INSTANCE(sycl::ref_layer_normalization_bwd_t)

        // PReLU
        INSTANCE(sycl::ref_prelu_fwd_t)
        INSTANCE(sycl::ref_prelu_bwd_t)

        // Pooling
        // INSTANCE(cnnl_pooling_fwd_t)
        // INSTANCE(cnnl_pooling_bwd_t)
        INSTANCE(sycl::ref_pooling_fwd_t)
        INSTANCE(sycl::ref_pooling_bwd_t)

        // LRN
        // INSTANCE(cnnl_lrn_fwd_t)
        // INSTANCE(cnnl_lrn_bwd_t)
        INSTANCE(sycl::ref_sycl_lrn_fwd_t)
        INSTANCE(sycl::ref_sycl_lrn_bwd_t)

        // Inner Product

        // Softmax
        INSTANCE(cnnl_softmax_fwd_t)
        INSTANCE(cnnl_softmax_bwd_t)
        INSTANCE(sycl::ref_sycl_softmax_fwd_t)
        INSTANCE(sycl::ref_sycl_softmax_bwd_t)

        // Binary
        // INSTANCE(cnnl_binary_t)
        INSTANCE(sycl::ref_binary_t)

        // MatMul
        // INSTANCE(cnnl_matmul_t)

        // Resampling
        INSTANCE(sycl::ref_resampling_fwd_t)
        INSTANCE(sycl::ref_resampling_bwd_t)

        // Reduction
        // INSTANCE(cnnl_reduction_t)

        // Shuffle
        INSTANCE(sycl::ref_shuffle_t)
        nullptr,
};
// clang-format on
} // namespace
const dnnl::impl::impl_list_item_t *sycl_bang_engine_t::get_implementation_list(
        const op_desc_t *) const {
    return sycl_bang_impl_list;
}

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl
