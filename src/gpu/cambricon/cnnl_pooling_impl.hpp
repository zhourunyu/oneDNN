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
            kernel_padding_[1] = static_cast<int>(pd->padB());
            kernel_padding_[2] = static_cast<int>(pd->padL());
            kernel_padding_[3] = static_cast<int>(pd->padR());

            kernel_strides_[0] = static_cast<int>(pd->KSH());
            kernel_strides_[1] = static_cast<int>(pd->KSW());
            kernel_dilations_[0] = static_cast<int>(pd->KDH() + 1);
            kernel_dilations_[1] = static_cast<int>(pd->KDW() + 1);
        } else {
            kernel_padding_[0] = static_cast<int>(pd->padFront());
            kernel_padding_[1] = static_cast<int>(pd->padBack());
            kernel_padding_[2] = static_cast<int>(pd->padT());
            kernel_padding_[3] = static_cast<int>(pd->padB());
            kernel_padding_[4] = static_cast<int>(pd->padL());
            kernel_padding_[5] = static_cast<int>(pd->padR());

            kernel_strides_[0] = static_cast<int>(pd->KSD());
            kernel_strides_[1] = static_cast<int>(pd->KSH());
            kernel_strides_[2] = static_cast<int>(pd->KSW());
            kernel_dilations_[0] = static_cast<int>(pd->KDD() + 1);
            kernel_dilations_[1] = static_cast<int>(pd->KDH() + 1);
            kernel_dilations_[2] = static_cast<int>(pd->KDW() + 1);
        }
        out_h_size_ = dims_[dst][ndims_ - 2];
        out_w_size_ = dims_[dst][ndims_ - 1];

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

        if (ndims_ == 4) {
            CHECK(CNNL_EXECUTE_FUNC_S(cnnlSetPooling2dDescriptor_v2, pool_desc_,
                    pool_mode_, CNNL_NOT_PROPAGATE_NAN,
                    kernel_dims_[0], kernel_dims_[1],
                    kernel_padding_[0], kernel_padding_[1],
                    kernel_padding_[2], kernel_padding_[3],
                    kernel_strides_[0], kernel_strides_[1],
                    kernel_dilations_[0], kernel_dilations_[1], /*ceil_mode*/false));
        } else {
            CHECK(CNNL_EXECUTE_FUNC_S(cnnlSetPoolingNdDescriptor_v2, pool_desc_,
                    pool_mode_, CNNL_NOT_PROPAGATE_NAN, ndims_, kernel_dims_,
                    kernel_padding_, kernel_strides_, kernel_dilations_, /*ceil_mode*/false));
        }

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
    int kernel_dilations_[DNNL_MAX_NDIMS];
    const float alpha_ = 1.f, beta_ = 0.f;
    int ndims_, kernel_ndims_;
    bool is_training_ = false;
    int out_h_size_, out_w_size_;
    std::size_t x_size_bytes_ = 0, y_size_bytes_ = 0;
};

struct cnnl_pooling_fwd_impl_t : public cnnl_pooling_impl_base_t {
    status_t init(const pooling_pd_t *pd) override {
        return cnnl_pooling_impl_base_t::init_common(pd);
    }

    void execute(cnnlHandle_t handle, void *x, void *y, void *ws_x,
            void *ws_y) const override {
        void *workspace = nullptr;
        size_t workspace_size = 0;
        CNNL_EXECUTE_FUNC(cnnlGetPoolingWorkspaceSize, handle, pool_mode_,
                out_w_size_, out_h_size_, &workspace_size);
        if (workspace_size > 0) {
            BANG_EXECUTE_FUNC(cnMalloc, (CNaddr *)&workspace, workspace_size);
        }

        void *extra_input_host = nullptr, *extra_input = nullptr;
        size_t extra_input_size = 0;
        CNNL_EXECUTE_FUNC(cnnlGetPoolingExtraInputSize, handle, pool_mode_,
            out_w_size_, out_h_size_, &extra_input_size);
        if (extra_input_size > 0) {
            extra_input_host = new int8_t[extra_input_size];
            CNNL_EXECUTE_FUNC(cnnlInitPoolingExtraInput, handle, pool_desc_,
                    tensor_descs_[src], tensor_descs_[dst], extra_input_host);
            BANG_EXECUTE_FUNC(cnMalloc, (CNaddr *)&extra_input, extra_input_size);
            BANG_EXECUTE_FUNC(cnMemcpy, (CNaddr)extra_input, (CNaddr)extra_input_host, extra_input_size);
        }

        CNNL_EXECUTE_FUNC(cnnlPoolingForward_v2, handle, pool_desc_, &alpha_,
                tensor_descs_[src], x, &beta_, extra_input, tensor_descs_[dst], y, workspace, workspace_size);

        if (is_training_) {
            // Copy x and y into workspace so that they can be used
            // in the backward pass
            BANG_EXECUTE_FUNC(cnMemcpy, (CNaddr)ws_x, (CNaddr)x, x_size_bytes_);
            BANG_EXECUTE_FUNC(cnMemcpy, (CNaddr)ws_y, (CNaddr)y, y_size_bytes_);
        }

        if (workspace) {
            BANG_EXECUTE_FUNC(cnFree, (CNaddr)workspace);
        }
        if (extra_input) {
            BANG_EXECUTE_FUNC(cnFree, (CNaddr)extra_input);
        }
        if (extra_input_host) {
            delete[] static_cast<int8_t *>(extra_input_host);
        }
    }
};

struct cnnl_pooling_bwd_impl_t : public cnnl_pooling_impl_base_t {
    status_t init(const pooling_pd_t *pd) override {
        return cnnl_pooling_impl_base_t::init_common(pd);
    }

    void execute(cnnlHandle_t handle, void *dx, void *dy, void *ws_x,
            void *ws_y) const override {
        // y not needed in max pooling backward
        auto y_desc = (pool_mode_ == CNNL_POOLING_MAX) ? nullptr : tensor_descs_[dst];
        void *y_ptr = (pool_mode_ == CNNL_POOLING_MAX) ? nullptr : ws_y;

        CNNL_EXECUTE_FUNC(cnnlPoolingBackward, handle, pool_desc_, &alpha_,
                y_desc, y_ptr, tensor_descs_[dst], dy,
                tensor_descs_[src], ws_x, &beta_, tensor_descs_[src], dx);
    }
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
