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

#ifndef GPU_CAMBRICON_CNNL_CONVOLUTION_IMPL_HPP
#define GPU_CAMBRICON_CNNL_CONVOLUTION_IMPL_HPP

#include "cnnl.h"

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/utils.hpp"
#include "gpu/cambricon/cnnl_conv_filter_adjustment_base.hpp"
#include "gpu/cambricon/cnnl_convolution_pd.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/cambricon/sycl_bang_scoped_context.hpp"
#include "gpu/cambricon/sycl_bang_stream.hpp"
#include "gpu/cambricon/sycl_bang_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct cnnl_convolution_impl_base_t
    : public cnnl_conv_filter_adjustment_base_t {
protected:
    enum io { x = 0, bias, weights, y, NUM_IO };
    memory_desc_t dnnl_descs[NUM_IO];
    cnnlConvolutionDescriptor_t conv_desc;
    int padding[CNNL_DIM_MAX];
    int dilation[CNNL_DIM_MAX];
    cnnlTensorDescriptor_t descs[NUM_IO];
    cnnlDataType_t data_types[NUM_IO];
    int ndims[NUM_IO];
    int dims[NUM_IO][DNNL_MAX_NDIMS];
    int filter_strides[DNNL_MAX_NDIMS];
    cnnlTensorLayout_t formats[NUM_IO];
    bool filter_needs_transform = false;
    float beta = 0.f;
    int group_count = 1;
    bool with_groups = false;
    size_t scratchpad_size = 0;
    bool with_bias = false;

    bool do_scaling = false;
    bool use_temp_dst_ = false;
    cnnlDataType_t computation_data_type = CNNL_DTYPE_FLOAT;
    cnnlDataType_t reorder_type = CNNL_DTYPE_INT8;

public:
    virtual ~cnnl_convolution_impl_base_t() {
        if (conv_desc)
            CNNL_EXECUTE_FUNC_V(cnnlDestroyConvolutionDescriptor, conv_desc);
        for (size_t i = 0; i < io::NUM_IO; i++) {
            if (descs[i]) {
                CNNL_EXECUTE_FUNC_V(cnnlDestroyTensorDescriptor, descs[i]);
            }
        }
    }
    virtual status_t configure_alg_kind(engine_t *, convolution_pd_t *pd) = 0;

    virtual bool supported_filter_format(
            const memory_desc_t *md) const override {
        const memory_desc_wrapper mem_wrapper(md);

        return with_groups ? mem_wrapper.matches_one_of_tag(
                            format_tag::gowi, format_tag::gohwi,
                            format_tag::godhwi)
                            : mem_wrapper.matches_one_of_tag(
                                    format_tag::owi, format_tag::ohwi,
                                    format_tag::odhwi);
    }

    bool using_transformed_filter() const { return filter_needs_transform; }
    bool with_scratchpad() const { return scratchpad_size > 0; }

    virtual status_t init(engine_t *engine, convolution_pd_t *pd,
            bool use_scratch_dst = false) {
        CHECK(configure_parameters(pd));
        CHECK(create_cnnl_descs(pd));
        CHECK(check_output_dims());
        CHECK(configure_alg_kind(engine, pd));
        CHECK(init_scratchpad(engine, pd));

        return status::success;
    }

    virtual status_t init_zero_dims(convolution_pd_t *pd) {
        return status::success;
    }
    void get_dims(int io) {
        convert_dims(
                dnnl_descs[io].dims, dims[io], dnnl_descs[io].ndims, ndims[io]);
        if (ndims[io] > dnnl_descs[io].ndims) {
            // [N, C, W, 1] -> [N, C, 1, W]
            // or [O, I, W, 1] -> [O, I, 1, W]
            std::swap(dims[io][ndims[io] - 1], dims[io][ndims[io] - 2]);
        }
        if (formats[io] == CNNL_LAYOUT_NHWC || formats[io] == CNNL_LAYOUT_HWCN) {
            transpose_dims(dims[io], ndims[io], formats[io]);
        }
    }
    status_t configure_parameters(const convolution_pd_t *pd) {
        if (pd->ndims() > CNNL_DIM_MAX) { return status::invalid_arguments; }
        CHECK(set_padding_and_dilation(pd));
        with_groups = pd->with_groups();
        with_bias = pd->with_bias();
        beta = 0.0f;
        do_scaling = !pd->attr()->scales_.has_default_values();
        dnnl_descs[x] = *pd->invariant_src_md();
        dnnl_descs[weights] = *pd->invariant_wei_md();
        dnnl_descs[y] = *pd->invariant_dst_md();
        if (with_bias) dnnl_descs[bias] = *pd->invariant_bia_md();
        if (with_groups) {
            group_count = pd->G();
        }

        ndims[x] = std::max(dnnl_descs[x].ndims, 4);
        ndims[weights] = std::max(dnnl_descs[weights].ndims, 4 + with_groups);
        ndims[y] = std::max(dnnl_descs[y].ndims, 4);

        CHECK(convert_data_type(&dnnl_descs[x], &data_types[x]));
        CHECK(convert_data_type(&dnnl_descs[weights], &data_types[weights]));
        CHECK(convert_data_type(&dnnl_descs[y], &data_types[y]));

        CHECK(get_formats());
        set_compute_format();
        get_dims(x);
        get_dims(weights);
        get_dims(y);

        if (!supported_filter_format(&dnnl_descs[weights])) {
            // we transform the filter based on src format
            CHECK(init_filter_transformation(data_types[weights],
                    ndims[weights], dims[weights], formats[weights], formats[x]));
            formats[weights] = formats[x];
            filter_needs_transform = true;
        } else {
            CHECK(get_filter_format());
        }
        // get new dims after filter transformation
        get_dims(weights);

        if (with_groups) {
            dims[weights][1] *= group_count;
            ndims[weights] = std::max(4, ndims[weights] - 1);
        }

        if (with_bias) {
            ndims[bias] = dnnl_descs[bias].ndims;
            assert(ndims[bias] == 1);
            CHECK(convert_data_type(&dnnl_descs[bias], &data_types[bias]));
            convert_dims(
                    dnnl_descs[bias].dims, dims[bias], ndims[bias]);
            formats[bias] = formats[y];
        }

        return status::success;
    }

    status_t create_cnnl_descs(const convolution_pd_t *pd) {
        CHECK(create_and_set_convolution_desc(pd));
        CHECK(create_and_set_tensor_descriptor_ex(
                &descs[x], formats[x], data_types[x], ndims[x], dims[x]));
        CHECK(create_and_set_tensor_descriptor_ex(&descs[weights], formats[weights],
                data_types[weights], ndims[weights],
                dims[weights] + with_groups));
        CHECK(create_and_set_tensor_descriptor_ex(
                &descs[y], formats[y], data_types[y], ndims[y], dims[y]));

        if (with_bias) {
            CHECK(create_and_set_tensor_descriptor_ex(&descs[bias],
                    formats[bias], data_types[bias], ndims[bias], dims[bias]));
        }

        return status::success;
    }
    virtual status_t init_scratchpad(engine_t *engine, convolution_pd_t *pd) {
        if (filter_needs_transform) {
            auto sz = memory_desc_wrapper(&dnnl_descs[weights]).size();
            auto data_size
                    = types::data_type_size(pd->invariant_wei_md(0)->data_type);
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_cnnl_filter, sz,
                    data_size);
        }
        return status::success;
    };

    status_t create_and_set_convolution_desc(const convolution_pd_t *pd) {
        CNNL_EXECUTE_FUNC_V(cnnlCreateConvolutionDescriptor, &conv_desc);
        CNNL_EXECUTE_FUNC_V(cnnlSetConvolutionDescriptor, conv_desc,
                ndims[x], padding, filter_strides, dilation,
                group_count, computation_data_type);
        return status::success;
    }

    status_t set_padding_and_dilation(const convolution_pd_t *pd) {
        int actual_ndims = pd->ndims();
        if (actual_ndims == 3) {
            padding[0] = 0;
            padding[1] = 0;
            padding[2] = static_cast<int>(pd->padL());
            padding[3] = static_cast<int>(pd->padR());
            dilation[0] = 1;
            dilation[1] = static_cast<int>(pd->KDW() + 1);

            filter_strides[0] = 1;
            filter_strides[1] = static_cast<int>(pd->KSW());
        } else if (actual_ndims == 4) {
            padding[0] = static_cast<int>(pd->padT());
            padding[1] = static_cast<int>(pd->padB());
            padding[2] = static_cast<int>(pd->padL());
            padding[3] = static_cast<int>(pd->padR());

            dilation[0] = static_cast<int>(pd->KDH() + 1);
            dilation[1] = static_cast<int>(pd->KDW() + 1);

            filter_strides[0] = static_cast<int>(pd->KSH());
            filter_strides[1] = static_cast<int>(pd->KSW());
        } else {
            padding[0] = static_cast<int>(pd->padFront());
            padding[1] = static_cast<int>(pd->padBack());
            padding[2] = static_cast<int>(pd->padT());
            padding[3] = static_cast<int>(pd->padB());
            padding[4] = static_cast<int>(pd->padL());
            padding[5] = static_cast<int>(pd->padR());

            dilation[0] = static_cast<int>(pd->KDD() + 1);
            dilation[1] = static_cast<int>(pd->KDH() + 1);
            dilation[2] = static_cast<int>(pd->KDW() + 1);

            filter_strides[0] = static_cast<int>(pd->KSD());
            filter_strides[1] = static_cast<int>(pd->KSH());
            filter_strides[2] = static_cast<int>(pd->KSW());
        }
        return status::success;
    }

    virtual void execute(
            cnnlHandle_t handle, const std::vector<void *> &args) const = 0;

    void execute_sum(cnnlHandle_t handle, void *x, void *y, float alpha_,
            float beta_) const {
        float alpha = alpha_;
        float beta = beta_;
        size_t workspace_size_sum = 0;
        void *workspace_sum = nullptr;
        CNNL_EXECUTE_FUNC_V(cnnlGetAssignAddWorkspaceSize, handle, descs[io::y], 
                descs[io::y], &workspace_size_sum);
        if (workspace_size_sum > 0) {
            BANG_EXECUTE_FUNC_V(cnMalloc, (CNaddr *)&workspace_sum, workspace_size_sum);
        }
        CNNL_EXECUTE_FUNC_V(cnnlAssignAdd, handle, &alpha, descs[io::y], x,
                workspace_sum, workspace_size_sum, &beta, descs[io::y], y);
        if (workspace_sum != nullptr) {
            BANG_EXECUTE_FUNC_V(cnFree, (CNaddr)workspace_sum);
        }
    }

    void execute_set_weights_bias(
            cnnlHandle_t handle, void *weights, void *bias, float value) {
        CNNL_EXECUTE_FUNC_V(cnnlFill_v3, handle, cnnlPointerMode_t::CNNL_POINTER_MODE_HOST, 
                &value, descs[io::weights], weights);
        if (bias) {
            CNNL_EXECUTE_FUNC_V(cnnlFill_v3, handle, cnnlPointerMode_t::CNNL_POINTER_MODE_HOST, 
                &value, descs[io::bias], bias);
        }
    }

    bool with_eltwise(const convolution_pd_t *pd, int position) const {
        return pd->attr()->post_ops_.contain(primitive_kind::eltwise, position);
    }

    status_t check_output_dims() const {
        int expected_dims[CNNL_DIM_MAX] = {};
        CNNL_EXECUTE_FUNC_V(cnnlGetConvolutionForwardOutputDim, conv_desc,
                descs[x], descs[weights], ndims[y], &expected_dims[0]);
        for (size_t i = 0; i < ndims[y]; i++) {
            if (dims[y][i] != expected_dims[i]) return status::unimplemented;
        }
        return status::success;
    }

    void set_compute_format() {
        switch (data_types[x]) {
            case CNNL_DTYPE_INT8:
                computation_data_type = CNNL_DTYPE_INT32;
                break;
            case CNNL_DTYPE_BFLOAT16:
                computation_data_type = CNNL_DTYPE_FLOAT;
                break;
            default: computation_data_type = data_types[y];
        }
    }

    status_t get_filter_format() {
        memory_desc_wrapper wrapper(&dnnl_descs[weights]);
        if ((!with_groups
                           && wrapper.matches_one_of_tag(format_tag::owi,
                                   format_tag::ohwi, format_tag::odhwi))
                || (with_groups
                        && wrapper.matches_one_of_tag(format_tag::gowi,
                                format_tag::gohwi, format_tag::godhwi))) {
            formats[weights] = cnnlTensorLayout_t::CNNL_LAYOUT_NHWC;
        } else {
            return status::unimplemented;
        }

        return status::success;
    }

    status_t get_formats() {
        CHECK(get_format(&dnnl_descs[x], formats[x]));
        CHECK(get_format(&dnnl_descs[y], formats[y]));
        if (formats[x] != CNNL_LAYOUT_NHWC || formats[y] != CNNL_LAYOUT_NHWC)
            return status::unimplemented;
        return status::success;
    }

    bool use_temp_dst() const { return use_temp_dst_; }
};

struct cnnl_convolution_impl_fwd_t : public cnnl_convolution_impl_base_t {
protected:
    cnnlActivationDescriptor_t eltwise_desc = nullptr;
    cnnlConvolutionForwardAlgo_t fwd_alg_kind;
    int requested_algo_count = 0;
    int returned_algo_count = 0;
    int num_post_ops = 0;
    primitive_kind_t post_ops[2];
    float sum_scale = 1.0f;
    bool conv_bias = false;

public:
    virtual ~cnnl_convolution_impl_fwd_t() {
        if (eltwise_desc)
            CNNL_EXECUTE_FUNC_V(
                    cnnlDestroyActivationDescriptor, eltwise_desc);
    }

    status_t configure_post_ops(convolution_pd_t *pd) {
        auto &p = pd->attr()->post_ops_;
        num_post_ops = p.len();
        for (size_t i = 0; i < p.len(); i++) {
            post_ops[i] = p.entry_[i].kind;
            if (post_ops[i] == dnnl_eltwise) {
                CHECK(create_and_set_eltwise_descriptor(pd));
            }
            if (post_ops[i] == dnnl_sum) { sum_scale = p.entry_[i].sum.scale; }
        }

        // Try to fuse kernels
        // pattern 1: conv + bias
        conv_bias = !do_scaling;

        return status::success;
    }

    status_t init(engine_t *engine, convolution_pd_t *pd,
            bool use_scratch_dst) override {
        use_temp_dst_ = use_scratch_dst;
        CHECK(configure_parameters(pd));
        CHECK(create_cnnl_descs(pd));
        CHECK(configure_alg_kind(engine, pd));
        CHECK(configure_post_ops(pd));
        CHECK(init_scratchpad(engine, pd));

        return status::success;
    }

    void execute_eltwise(cnnlHandle_t handle, void *src, void *dst) const {
        float alpha = 1.0f;
        float beta = 0.0f;
        CNNL_EXECUTE_FUNC_V(cnnlActivationForward, handle, eltwise_desc,
                &alpha, descs[io::y], src, &beta, descs[io::y], dst);
    }

    void execute(cnnlHandle_t handle,
            const std::vector<void *> &args) const override {
        auto x = args[0], weights = args[1], y = args[2], bias = args[3],
             scratchpad = args[4], post_op_scratch = args[6],
             src_scale = args[7], wei_scale = args[8], dst_scale = args[9];
        void *output = use_temp_dst_ ? post_op_scratch : y;
        if (using_transformed_filter()) {
            auto w_scratch = args[5];
            transform_filter(handle, weights, w_scratch);
            weights = w_scratch;
        }

        bool fused = conv_bias;

        float scale = 1.0f;
        if (src_scale || wei_scale) {
            if (src_scale) {
                float host_src_scale = 1.0f;
                BANG_EXECUTE_FUNC(cnMemcpy, (CNaddr)&host_src_scale,
                        (CNaddr)src_scale, sizeof(float));
                scale *= host_src_scale;
            }
            if (wei_scale) {
                float host_wei_scale = 1.0f;
                BANG_EXECUTE_FUNC(cnMemcpy, (CNaddr)&host_wei_scale,
                        (CNaddr)wei_scale, sizeof(float));
                scale *= host_wei_scale;
            }
        }

        if (fused) {
            // no scaling when fused
            assert(scale == 1.0f && beta == 0.0f);
            CNNL_EXECUTE_FUNC_V(cnnlConvolutionForward, handle, conv_desc,
                    fwd_alg_kind, nullptr, descs[io::x], x, 
                    descs[io::weights], weights, descs[io::bias], 
                    bias, scratchpad, scratchpad_size, nullptr,
                    descs[io::y], output);
        } else {
            float bias_beta = 1.0f;
            CNNL_EXECUTE_FUNC_V(cnnlConvolutionForward, handle, conv_desc,
                    fwd_alg_kind, nullptr, descs[io::x], x, 
                    descs[io::weights], weights, nullptr, nullptr,
                    scratchpad, scratchpad_size, nullptr,
                    descs[io::y], output);

            size_t workspace_size_add = 0;
            void *workspace_add = nullptr;
            CNNL_EXECUTE_FUNC_V(cnnlGetBiasAddWorkspaceSize, handle,
                    descs[io::bias], descs[io::y], &workspace_size_add);
            if (workspace_size_add > 0) {
                BANG_EXECUTE_FUNC(cnMalloc, (CNaddr *)&workspace_add,
                        workspace_size_add);
            }
            CNNL_EXECUTE_FUNC_V(cnnlBiasAdd, handle, &bias_beta,
                    descs[io::bias], bias, workspace_add, 
                    workspace_size_add, &scale, descs[io::y], output);
            if (workspace_add != nullptr) {
                BANG_EXECUTE_FUNC(cnFree, (CNaddr)workspace_add);
            }
        }
        for (int i = 0; i < num_post_ops; i++) {
            bool last_op = i == num_post_ops - 1;
            switch (post_ops[i]) {
                case dnnl_sum:
                    if (last_op) {
                        execute_sum(
                                handle, post_op_scratch, y, 1.0f, sum_scale);
                    } else {
                        execute_sum(
                                handle, y, post_op_scratch, sum_scale, 1.0f);
                    }
                    break;

                case dnnl_eltwise:
                    if (last_op) {
                        execute_eltwise(handle, output, y);
                    } else {
                        execute_eltwise(handle, output, post_op_scratch);
                    }
                    break;
                default: assert(!"unsupported post op");
            }
        }

        if (dst_scale) {
            float host_dst_scale = 1.0f;
            BANG_EXECUTE_FUNC(cnMemcpy, (CNaddr)&host_dst_scale,
                    (CNaddr)dst_scale, sizeof(float));
            float inv_scale = 1.0f / host_dst_scale;
            size_t workspace_size_add = 0;
            void *workspace_add = nullptr;
            CNNL_EXECUTE_FUNC_V(cnnlGetBiasAddWorkspaceSize, handle,
                    nullptr, descs[io::y], &workspace_size_add);
            if (workspace_size_add > 0) {
                BANG_EXECUTE_FUNC(cnMalloc, (CNaddr *)&workspace_add,
                        workspace_size_add);
            }
            float bias_beta = 0.0f;
            CNNL_EXECUTE_FUNC(cnnlBiasAdd, handle, &bias_beta, nullptr, 
                    nullptr, workspace_add, workspace_size_add, &inv_scale, 
                    descs[io::y], y);
        }
    }
    status_t init_scratchpad(engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto bang_stream
                = utils::downcast<sycl_bang_stream_t *>(service_stream);
        auto handle = bang_stream->get_cnnl_handle();

        CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetConvolutionForwardWorkspaceSize,
                handle, descs[x], descs[weights], descs[y], descs[bias], 
                conv_desc, fwd_alg_kind, &scratchpad_size));
        if (scratchpad_size > 0)
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_cnnl_algo,
                    scratchpad_size, size_t(1));

        return cnnl_convolution_impl_base_t::init_scratchpad(engine, pd);
    }
    status_t configure_alg_kind(
            engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(engine);
        bang_sycl_scoped_context_handler_t sc(sycl_engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto bang_stream
                = utils::downcast<sycl_bang_stream_t *>(service_stream);
        auto handle = bang_stream->get_cnnl_handle();

        CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetConvolutionForwardAlgorithm, handle,
                conv_desc, descs[x], descs[weights], descs[y],
                cnnlConvolutionFwdPreference_t::CNNL_CONVOLUTION_FWD_FASTEST, &fwd_alg_kind));

        return status::success;
    }

    status_t create_and_set_eltwise_descriptor(const convolution_pd_t *pd) {

        CHECK(CNNL_EXECUTE_FUNC_S(
                cnnlCreateActivationDescriptor, &eltwise_desc));

        cnnlActivationMode_t act_mode;
        switch (eltwise_algorithm_kind(pd)) {
            case alg_kind::eltwise_tanh:
                act_mode = CNNL_ACTIVATION_TANH;
                break;
            case alg_kind::eltwise_elu: act_mode = CNNL_ACTIVATION_ELU; break;
            case alg_kind::eltwise_relu:
                act_mode = CNNL_ACTIVATION_RELU;
                break;
            case alg_kind::eltwise_logistic:
                act_mode = CNNL_ACTIVATION_SIGMOID;
                break;
            default: return status::unimplemented;
        }
        CHECK(CNNL_EXECUTE_FUNC_S(cnnlSetActivationDescriptor, eltwise_desc,
                act_mode, cnnlNanPropagation_t::CNNL_NOT_PROPAGATE_NAN,
                eltwise_alpha(pd)));

        return status::success;
    }

    dnnl::impl::alg_kind_t eltwise_algorithm_kind(
            const convolution_pd_t *pd) const {
        const int eltwise_idx
                = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        return pd->attr()->post_ops_.entry_[eltwise_idx].eltwise.alg;
    }

    float eltwise_alpha(const convolution_pd_t *pd) const {
        const int eltwise_idx
                = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        return pd->attr()->post_ops_.entry_[eltwise_idx].eltwise.alpha;
    }
};

struct cnnl_convolution_impl_bwd_data_t
    : public cnnl_convolution_impl_base_t {
protected:
    cnnlConvolutionBwdDataAlgo_t bwd_algo = CNNL_CONVOLUTION_BWD_DATA_ALGO_DIRECT;
    status_t configure_alg_kind(
            engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(engine);
        bang_sycl_scoped_context_handler_t sc(sycl_engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto bang_stream
                = utils::downcast<sycl_bang_stream_t *>(service_stream);
        auto handle = bang_stream->get_cnnl_handle();

        CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetConvolutionBackwardDataAlgorithm,
                handle, descs[weights], descs[y], conv_desc, descs[x],
                cnnlConvolutionBwdDataPreference_t::CNNL_CONVOLUTION_BWD_DATA_FASTEST, 
                &bwd_algo));

        return status::success;
    }

    status_t init_scratchpad(engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto bang_stream
                = utils::downcast<sycl_bang_stream_t *>(service_stream);
        auto handle = bang_stream->get_cnnl_handle();

        CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetConvolutionBackwardDataWorkspaceSize,
                handle, descs[io::weights], descs[io::y], conv_desc, descs[io::x],
                bwd_algo, &scratchpad_size));
        if (scratchpad_size > 0)
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_cnnl_algo,
                    scratchpad_size, size_t(1));

        return cnnl_convolution_impl_base_t::init_scratchpad(engine, pd);
    }

    void execute(cnnlHandle_t handle,
            const std::vector<void *> &args) const override {
        auto x = args[0], weights = args[1], y = args[2], bias = args[3],
             scratchpad = args[4];
        if (using_transformed_filter()) {
            auto w_scratch = args[5];
            transform_filter(handle, weights, w_scratch);
            weights = w_scratch;
        }
        const float bias_alpha = 1.0f;
        const float bias_beta = 1.0f;
        CNNL_EXECUTE_FUNC_V(cnnlConvolutionBackwardData, handle, nullptr,
                descs[io::weights], weights, descs[io::y], y, conv_desc, bwd_algo,
                scratchpad, scratchpad_size, nullptr, descs[io::x], x);
        if (with_bias) {
            size_t workspace_size_add = 0;
            void *workspace_add = nullptr;
            CNNL_EXECUTE_FUNC_V(cnnlBiasAdd, handle, &bias_alpha,
                    descs[io::bias], bias, workspace_add, workspace_size_add, 
                    &bias_beta, descs[io::x], x);
        }
    }
};

struct cnnl_convolution_impl_bwd_weights_t
    : public cnnl_convolution_impl_base_t {
protected:
    cnnlConvolutionBwdFilterAlgo_t bwd_filter_algo
            = CNNL_CONVOLUTION_BWD_FILTER_ALGO_DIRECT;

public:
    status_t init_zero_dims(convolution_pd_t *pd) override {
        if (pd->ndims() > CNNL_DIM_MAX) { return status::invalid_arguments; }
        dnnl_descs[weights] = *pd->invariant_wei_md();
        CHECK(get_format(&dnnl_descs[weights], formats[weights], true));
        ndims[y] = pd->invariant_dst_md()->ndims;
        ndims[weights] = dnnl_descs[weights].ndims - pd->with_groups();
        CHECK(convert_data_type(&dnnl_descs[weights], &data_types[weights]));
        convert_dims(dnnl_descs[weights].dims + pd->with_groups(),
                dims[weights], ndims[weights]);
        ndims[weights] = std::max(4, ndims[weights]);
        CHECK(create_and_set_tensor_descriptor_ex(&descs[weights],
                formats[weights], data_types[weights], ndims[weights], dims[weights]));

        if (pd->with_bias()) {
            return status::unimplemented;
            // dnnl_descs[bias] = *pd->invariant_bia_md();
            // ndims[bias] = dnnl_descs[bias].ndims;
            // assert(ndims[bias] == 1);
            // CHECK(convert_data_type(&dnnl_descs[bias], &data_types[bias]));
            // convert_dims(dnnl_descs[bias].padded_dims, dims[bias], ndims[bias],
            //         ndims[y]);
            // formats[bias] = formats[y];
            // CHECK(create_and_set_tensor_descriptor_ex(&descs[bias],
            //         formats[bias], data_types[bias], ndims[bias], dims[bias]));
        }
        return status::success;
    }
    virtual status_t configure_alg_kind(
            engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(engine);
        bang_sycl_scoped_context_handler_t sc(sycl_engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto bang_stream
                = utils::downcast<sycl_bang_stream_t *>(service_stream);
        auto handle = bang_stream->get_cnnl_handle();

        CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetConvolutionBackwardFilterAlgorithm,
                handle, conv_desc, descs[x], descs[y], descs[weights],
                cnnlConvolutionBwdFilterPreference_t::CNNL_CONVOLUTION_BWD_FILTER_FASTEST,
                &bwd_filter_algo));

        return status::success;
    }

    status_t init_scratchpad(engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto bang_stream
                = utils::downcast<sycl_bang_stream_t *>(service_stream);
        auto handle = bang_stream->get_cnnl_handle();

        CHECK(CNNL_EXECUTE_FUNC_S(
                cnnlGetConvolutionBackwardFilterWorkspaceSize, handle,
                descs[io::x], descs[io::y], descs[io::weights], conv_desc,
                bwd_filter_algo, &scratchpad_size));
        if (scratchpad_size > 0)
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_cnnl_algo,
                    scratchpad_size, size_t(1));

        return cnnl_convolution_impl_base_t::init_scratchpad(engine, pd);
    }

    void execute(cnnlHandle_t handle,
            const std::vector<void *> &args) const override {
        auto x = args[0], weights = args[1], y = args[2], bias = args[3],
             scratchpad = args[4];
        auto filter = weights;
        if (using_transformed_filter()) {
            auto w_scratch = args[5];
            transform_filter(handle, weights, w_scratch);
            filter = w_scratch;
        }
        const float alpha = 1.0f;
        const float bias_alpha = 1.0f;
        const float bias_beta = 0.0f;
        CNNL_EXECUTE_FUNC_V(cnnlConvolutionBackwardFilter, handle, nullptr,
                descs[io::x], x, descs[io::y], y, conv_desc, bwd_filter_algo,
                scratchpad, scratchpad_size, nullptr, descs[io::weights], filter);
        // if (with_bias) {
        //     CNNL_EXECUTE_FUNC_V(cnnlConvolutionBackwardBias, handle,
        //             &bias_alpha, descs[io::y], y, &bias_beta, descs[io::bias],
        //             bias);
        // }
        if (using_transformed_filter()) {
            undo_transform_filter(handle, filter, weights);
        }
    }
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
