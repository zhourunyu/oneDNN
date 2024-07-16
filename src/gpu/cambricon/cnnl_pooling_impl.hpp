/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef GPU_CAMBRICON_CNNL_POOLING_IMPL_HPP
#define GPU_CAMBRICON_CNNL_POOLING_IMPL_HPP

#include <cnnl.h>

#include "gpu/cambricon/sycl_bang_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct cnnl_pooling_impl_base_t {
    virtual status_t init(const pooling_pd_t *pd) = 0;

    virtual ~cnnl_pooling_impl_base_t() {
        for (size_t i = 0; i < NUM_IO; ++i) {
            if (tensor_descs_[i]) {
                CNNL_EXECUTE_FUNC_V(
                        cnnlDestroyTensorDescriptor, tensor_descs_[i]);
            }
        }

        if (pool_desc_) {
            CNNL_EXECUTE_FUNC_V(cnnlDestroyPoolingDescriptor, pool_desc_);
        }
    }

    virtual void execute(cnnlHandle_t handle, void *x, void *y, void *ws_x,
            void *ws_y) const = 0;

protected:
    status_t init_common(const pooling_pd_t *pd) {
        ndims_ = std::max(4, pd->ndims());
        kernel_ndims_ = ndims_ - 2;

        // Only 1D, 2D and 3D pooling is supported by cnnl
        if (kernel_ndims_ > 3) { return status::unimplemented; }

        // cnnl requires symmetric padding, however it seems that
        // configurations where padding in the beginning > padding at the end of
        // dimensions work as expected. When padding at the end of any dimension
        // > padding in the beginning of that dimension the results are wrong
        // since the data is rearranged incorrectly due to the limitation that
        // padding has to be the same. This applies to configurations which use
        // the "average include padding" algorithm. Therefore, such
        // configurations return status::unimplemented since the results are
        // wrong.
        if (pd->desc()->alg_kind == alg_kind::pooling_avg_include_padding
                && (pd->padL() < pd->padR() || pd->padT() < pd->padB()
                        || pd->padFront() < pd->padBack())) {
            return status::unimplemented;
        }

        is_training_ = pd->desc()->prop_kind == prop_kind::forward_training;
        bool is_fwd = pd->is_fwd();
        auto src_md = is_fwd ? pd->src_md() : pd->diff_src_md();
        auto dst_md = is_fwd ? pd->dst_md() : pd->diff_dst_md();

        if (has_zero_dims(src_md->dims, pd->ndims())
                || has_zero_dims(dst_md->dims, pd->ndims())) {
            return status::success;
        }

        if (is_training_) {
            auto src_wrap = memory_desc_wrapper(src_md);
            auto dst_wrap = memory_desc_wrapper(dst_md);
            x_size_bytes_ = src_wrap.size();
            y_size_bytes_ = dst_wrap.size();
        }

        convert_dims(src_md->padded_dims, dims_[src], pd->ndims());
        convert_dims(dst_md->padded_dims, dims_[dst], pd->ndims());

        convert_dims(pd->desc()->kernel, kernel_dims_, kernel_ndims_);

        // If 1D pooling
        if (pd->ndims() == 3) {
            // Convert to [n, c, 1, w] since the current format is
            // [n, c, w, 1]
            dims_[src][3] = dims_[src][2];
            dims_[src][2] = 1;

            dims_[dst][3] = dims_[dst][2];
            dims_[dst][2] = 1;

            // Set kernel dimensions to [1, kw]
            kernel_dims_[1] = kernel_dims_[0];
            kernel_dims_[0] = 1;
        }

        if (ndims_ == 4) {
            kernel_padding_[0] = static_cast<int>(pd->padT());
            kernel_padding_[1] = static_cast<int>(pd->padL());

            kernel_strides_[0] = static_cast<int>(pd->KSH());
            kernel_strides_[1] = static_cast<int>(pd->KSW());
        } else {
            kernel_padding_[0] = static_cast<int>(pd->padFront());
            kernel_padding_[1] = static_cast<int>(pd->padT());
            kernel_padding_[2] = static_cast<int>(pd->padL());

            kernel_strides_[0] = static_cast<int>(pd->KSD());
            kernel_strides_[1] = static_cast<int>(pd->KSH());
            kernel_strides_[2] = static_cast<int>(pd->KSW());
        }

        CHECK(convert_data_type(src_md, &data_types_[src]));
        CHECK(convert_data_type(dst_md, &data_types_[dst]));

        CHECK(convert_alg_kind(pd->desc()->alg_kind, &pool_mode_));

        cnnlTensorLayout_t src_format, dst_format;
        CHECK(get_format(src_md, src_format));
        CHECK(get_format(dst_md, dst_format));

        CHECK(transpose_dims(dims_[src], pd->ndims(), src_format));
        CHECK(transpose_dims(dims_[dst], pd->ndims(), dst_format));

        CHECK(create_and_set_tensor_descriptor_ex(&tensor_descs_[src],
                src_format, data_types_[src], ndims_, dims_[src]));
        CHECK(create_and_set_tensor_descriptor_ex(&tensor_descs_[dst],
                dst_format, data_types_[dst], ndims_, dims_[dst]));

        CHECK(create_and_set_pooling_descriptor(pd));

        return status::success;
    }

    status_t create_and_set_pooling_descriptor(const pooling_pd_t *pd) {
        CHECK(CNNL_EXECUTE_FUNC_S(cnnlCreatePoolingDescriptor, &pool_desc_));

        CHECK(CNNL_EXECUTE_FUNC_S(cnnlSetPoolingNdDescriptor, pool_desc_,
                pool_mode_, CNNL_PROPAGATE_NAN, ndims_, kernel_dims_,
                kernel_padding_, kernel_strides_));

        return status::success;
    }

    status_t convert_alg_kind(
            alg_kind_t alg_kind, cnnlPoolingMode_t *cnnl_alg_kind) const {
        switch (alg_kind) {
            case alg_kind::pooling_max:
                *cnnl_alg_kind = CNNL_POOLING_MAX;
                break;
            case alg_kind::pooling_avg_include_padding:
                *cnnl_alg_kind = CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
                break;
            case alg_kind::pooling_avg_exclude_padding:
                *cnnl_alg_kind = CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
                break;
            default: return status::unimplemented;
        }

        return status::success;
    }

    enum io { src = 0, dst, NUM_IO };
    cnnlDataType_t data_types_[NUM_IO];
    cnnlTensorDescriptor_t tensor_descs_[NUM_IO] = {};
    cnnlPoolingDescriptor_t pool_desc_;
    cnnlPoolingMode_t pool_mode_ = CNNL_POOLING_MAX;
    int dims_[NUM_IO][DNNL_MAX_NDIMS];
    int kernel_dims_[DNNL_MAX_NDIMS];
    int kernel_padding_[DNNL_MAX_NDIMS];
    int kernel_strides_[DNNL_MAX_NDIMS];
    const float alpha_ = 1.f, beta_ = 0.f;
    int ndims_, kernel_ndims_;
    bool is_training_ = false;
    std::size_t x_size_bytes_ = 0, y_size_bytes_ = 0;
};

struct cnnl_pooling_fwd_impl_t : public cnnl_pooling_impl_base_t {
    status_t init(const pooling_pd_t *pd) override {
        return cnnl_pooling_impl_base_t::init_common(pd);
    }

    void execute(cnnlHandle_t handle, void *x, void *y, void *ws_x,
            void *ws_y) const override {
        cnrtQueue_t queue;
        CNRT_CHECK(cnrtQueueCreate(&queue));
        CNNL_EXECUTE_FUNC_V(cnnlSetQueue, handle, queue);

        CNNL_EXECUTE_FUNC(cnnlGetPoolingWorkspaceSize, handle, pool_mode_, 
                dims_[dst][ndims_ - 2], dims_[dst][ndims_ - 1], &workspace_size_);
        if (workspace_size_ > 0) {
            BANG_EXECUTE_FUNC(cnMalloc, (CNaddr *)&workspace_, workspace_size_);
        }

        CNNL_EXECUTE_FUNC(cnnlPoolingForward, handle, pool_desc_, &alpha_,
                tensor_descs_[src], x, &beta_, tensor_descs_[dst], y, workspace_, workspace_size_);

        if (is_training_) {
            // Copy x and y into workspace so that they can be used
            // in the backward pass
            BANG_EXECUTE_FUNC(cnMemcpy, (CNaddr)ws_x, (CNaddr)x, x_size_bytes_);
            BANG_EXECUTE_FUNC(cnMemcpy, (CNaddr)ws_y, (CNaddr)y, y_size_bytes_);
        }

        CNRT_CHECK(cnrtQueueSync(queue));
        if (workspace_) {
            BANG_EXECUTE_FUNC(cnFree, (CNaddr)workspace_);
            workspace_ = nullptr;
        }
        CNRT_CHECK(cnrtQueueDestroy(queue));
    }

protected:
    mutable void *workspace_ = nullptr;
    mutable size_t workspace_size_ = 0;
};

struct cnnl_pooling_bwd_impl_t : public cnnl_pooling_impl_base_t {
    status_t init(const pooling_pd_t *pd) override {
        return cnnl_pooling_impl_base_t::init_common(pd);
    }

    void execute(cnnlHandle_t handle, void *dx, void *dy, void *ws_x,
            void *ws_y) const override {

        CNNL_EXECUTE_FUNC(cnnlPoolingBackward, handle, pool_desc_, &alpha_,
                tensor_descs_[dst], ws_y, tensor_descs_[dst], dy,
                tensor_descs_[src], ws_x, &beta_, tensor_descs_[src], dx);
    }
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
