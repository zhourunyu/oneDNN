/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef GPU_CAMBRICON_SYCL_BANG_ELTWISE_IMPL_HPP
#define GPU_CAMBRICON_SYCL_BANG_ELTWISE_IMPL_HPP

#include "cnnl.h"

#include "gpu/nvidia/sycl_bang_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cnnl_eltwise_impl_base_t {

public:
    virtual void execute(cnnlHandle_t handle, void **x, int size) const = 0;

    virtual status_t create_and_set_act_descriptor() {
        CHECK(CNNL_EXECUTE_FUNC_S(
                cnnlCreateActivationDescriptor, &act_desc_));

        CHECK(CNNL_EXECUTE_FUNC_S(cnnlSetActivationDescriptor, act_desc_,
                alg_kind, cnnlNanPropagation_t::CNNL_NOT_PROPAGATE_NAN, coef));

        return status::success;
    }

    // Mapping between dnnl algorithm and cnnl activation mode
    status_t convert_alg_kind(
            alg_kind_t alg_kind, cnnlActivationMode_t *bang_alg_kind) const {
        switch (alg_kind) {
            case alg_kind::eltwise_relu:
                *bang_alg_kind = cnnlActivationMode_t::CNNL_ACTIVATION_RELU;
                break;
            case alg_kind::eltwise_tanh:
                *bang_alg_kind = cnnlActivationMode_t::CNNL_ACTIVATION_TANH;
                break;
            case alg_kind::eltwise_elu:
                *bang_alg_kind = cnnlActivationMode_t::CNNL_ACTIVATION_ELU;
                break;
            case alg_kind::eltwise_logistic:
                *bang_alg_kind
                        = cnnlActivationMode_t::CNNL_ACTIVATION_SIGMOID;
                break;
            default: return status::unimplemented;
        }
        return status::success;
    }

    virtual ~cnnl_eltwise_impl_base_t() {
        if (act_desc_) {
            CNNL_EXECUTE_FUNC_V(cnnlDestroyActivationDescriptor, act_desc_);
        }
    }

protected:
    int ndims;
    cnnlActivationDescriptor_t act_desc_ = nullptr;
    cnnlActivationMode_t alg_kind;
    // alpha and beta are post operation scaling parameters used by cuDNN
    float alpha = 1;
    float beta = 0;
    // coef in cuDNN is use for Relu (is equal to zero)
    double coef = 0;
};

struct cnnl_eltwise_fwd_impl_t : public cnnl_eltwise_impl_base_t {
public:
    status_t init(const eltwise_fwd_pd_t *pd) {
        // If any of the dimensions are 0 we should not continue with creating
        // cnnl descriptors
        if (has_zero_dims(pd->src_md()->dims, pd->ndims())) {
            return status::success;
        }
        if (pd->ndims() > CNNL_DIM_MAX) { return status::invalid_arguments; }
        ndims = pd->ndims() < 4 ? 4 : pd->ndims();

        // Obtain source and destination dimensions, strides and datatype
        convert_dims(pd->src_md()->padded_dims, dims_, pd->ndims());
        convert_dims(pd->src_md()->format_desc.blocking.strides, strides_,
                pd->ndims());
        CHECK(convert_data_type(pd->src_md(), &data_type_));

        // Get cuDNN activation mode
        alg_kind_t alg = pd->desc()->alg_kind;
        CHECK(convert_alg_kind(alg, &alg_kind));
        coef = pd->desc()->alpha;

        CHECK(create_and_set_tensor_descriptor(
                &tensor_desc_, data_type_, ndims, dims_, strides_));
        CHECK(create_and_set_act_descriptor());
        return status::success;
    }

    void execute(cnnlHandle_t handle, void **x, int size) const override {
        // Confirm that 2 arguments were passed src and dst
        assert(size == 2);
        CNNL_EXECUTE_FUNC(cnnlActivationForward, handle, act_desc_, &alpha,
                tensor_desc_, x[0], &beta, tensor_desc_, x[1]);
    }

    ~cnnl_eltwise_fwd_impl_t() {
        CNNL_EXECUTE_FUNC_V(cnnlDestroyTensorDescriptor, tensor_desc_);
    }

private:
    int strides_[DNNL_MAX_NDIMS];
    int dims_[DNNL_MAX_NDIMS];
    cnnlDataType_t data_type_;
    cnnlTensorDescriptor_t tensor_desc_;
};

struct cnnl_eltwise_bwd_impl_t : public cnnl_eltwise_impl_base_t {

public:
    status_t init(const eltwise_bwd_pd_t *pd) {
        // If any of the dimensions are 0 we should not continue with creating
        // cnnl descriptors
        if (memory_desc_wrapper(pd->data_md()).has_zero_dim())
            return status::success;

        if (pd->ndims() > CNNL_DIM_MAX) { return status::invalid_arguments; }
        ndims = pd->ndims() < 4 ? 4 : pd->ndims();

        // Obtain dimension and strides for the backward eltwise operation
        convert_dims(pd->src_md()->padded_dims, dims_, pd->ndims());

        convert_dims(pd->src_md()->format_desc.blocking.strides, strides_,
                pd->ndims());

        alg_kind_t alg = pd->desc()->alg_kind;
        CHECK(convert_alg_kind(alg, &alg_kind));
        coef = pd->desc()->alpha;

        // Check validity of input
        assert(pd->diff_dst_md()->data_type == pd->src_md()->data_type);
        assert(pd->diff_dst_md()->data_type == pd->diff_src_md()->data_type);

        CHECK(convert_data_type(pd->src_md(), &data_type_));

        CHECK(create_and_set_tensor_descriptor(
                &tensor_desc_src_, data_type_, ndims, dims_, strides_));
        CHECK(create_and_set_tensor_descriptor(
                &tensor_diff_desc_, data_type_, ndims, dims_, strides_));
        CHECK(create_and_set_act_descriptor());
        return status::success;
    }

    void execute(cnnlHandle_t handle, void **x, int size) const override {
        // Assert that 3 arguments were passed src, diff_dst and diff_src
        assert(size == 3);
        void *dy = x[1];
        void *dx = x[2];
        CNNL_EXECUTE_FUNC(cnnlActivationBackward, handle, act_desc_, &alpha,
                tensor_desc_src_, x[0], tensor_diff_desc_, dy, tensor_desc_src_,
                x[0], &beta, tensor_diff_desc_, dx);
    }

    ~cnnl_eltwise_bwd_impl_t() {
        CNNL_EXECUTE_FUNC_V(cnnlDestroyTensorDescriptor, tensor_desc_src_);
        CNNL_EXECUTE_FUNC_V(cnnlDestroyTensorDescriptor, tensor_diff_desc_);
    }

private:
    int dims_[DNNL_MAX_NDIMS];
    int strides_[DNNL_MAX_NDIMS];
    cnnlTensorDescriptor_t tensor_diff_desc_;
    cnnlDataType_t data_type_;
    cnnlTensorDescriptor_t tensor_desc_src_;
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
