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

#ifndef GPU_CAMBRICON_CNNL_MATMUL_IMPL_HPP
#define GPU_CAMBRICON_CNNL_MATMUL_IMPL_HPP

#include "cnnl.h"

#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/cambricon/sycl_bang_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct cnnl_matmul_impl_t {

    bool with_eltwise(int position, const matmul_pd_t *pd) const {
        return pd->attr()->post_ops_.contain(primitive_kind::eltwise, position);
    }

    float eltwise_alpha(const matmul_pd_t *pd) const {
        int eltwise_idx_ = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        return with_eltwise(0, pd) || with_eltwise(1, pd)
                ? pd->attr()->post_ops_.entry_[eltwise_idx_].eltwise.alpha
                : 1.0f;
    }

    float eltwise_beta(const matmul_pd_t *pd) const {
        int eltwise_idx_ = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        return with_eltwise(0, pd) || with_eltwise(1, pd)
                ? pd->attr()->post_ops_.entry_[eltwise_idx_].eltwise.beta
                : 0.0f;
    }

    alg_kind_t eltwise_algo(const matmul_pd_t *pd) const {
        int eltwise_idx_ = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        return with_eltwise(0, pd) || with_eltwise(1, pd)
                ? pd->attr()->post_ops_.entry_[eltwise_idx_].eltwise.alg
                : dnnl_alg_kind_undef;
    }

    bool with_sum(const matmul_pd_t *pd) const {
        return pd->attr()->post_ops_.contain(primitive_kind::sum, 0)
                || pd->attr()->post_ops_.contain(primitive_kind::sum, 1);
    }

    // Returns scaling factor for post-ops=sum operation
    float sum_scale(const matmul_pd_t *pd) const {
        int sum_idx_ = pd->attr()->post_ops_.find(primitive_kind::sum);
        return pd->attr()->post_ops_.entry_[sum_idx_].sum.scale;
    }

    // creates operation descriptor based on the elemen-wise operation specified
    status_t create_and_set_op_descriptor(const matmul_pd_t *pd) {
        CHECK(CNNL_EXECUTE_FUNC_S(
                cnnlCreateActivationDescriptor, &act_desc_));

        cnnlActivationMode_t mode;
        cnnlActivationPreference_t prefer = CNNL_ACTIVATION_HIGH_PRECISION;
        float coef = 1.0f, gamma = 1.0f, scale = 1.0f;

        // Used in leakyrelu and elu
        float alpha = eltwise_alpha(pd);

        switch (eltwise_algo(pd)) {
            case alg_kind::eltwise_relu:
                if (alpha == 0.0f)
                    mode = cnnlActivationMode_t::CNNL_ACTIVATION_RELU;
                else {
                    mode = cnnlActivationMode_t::CNNL_ACTIVATION_LEAKYRELU;
                    coef = alpha;
                }
                break;
            case alg_kind::eltwise_tanh:
                mode = cnnlActivationMode_t::CNNL_ACTIVATION_TANH;
                break;
            case alg_kind::eltwise_elu:
                mode = cnnlActivationMode_t::CNNL_ACTIVATION_ELU;
                gamma = alpha;
                break;
            case alg_kind::eltwise_logistic:
                mode = cnnlActivationMode_t::CNNL_ACTIVATION_SIGMOID;
                break;
            default: return status::unimplemented;
        }

        // NaNs by default are propagated in oneDNN, although the forward
        // convolution routine does not support this.
        auto propagate_nan = cnnlNanPropagation_t::CNNL_NOT_PROPAGATE_NAN;

        CHECK(CNNL_EXECUTE_FUNC_S(cnnlSetActivationDescriptor_v5, act_desc_,
                mode, prefer, propagate_nan, coef, /*sliced_dim*/ 0, gamma, 
                scale, /*is_result*/ true));

        return status::success;
    }

    status_t init(matmul_pd_t *pd) {
        isbatched_ = pd->batched();

        memory_desc_wrapper src_d = memory_desc_wrapper(pd->src_md());
        memory_desc_wrapper weights_d = memory_desc_wrapper(pd->weights_md());
        memory_desc_wrapper dst_d = memory_desc_wrapper(pd->dst_md());

        with_bias_ = pd->with_bias();
        if ((with_bias_)
                && (pd->weights_md(1)->data_type != pd->dst_md()->data_type)) {
            return status::unimplemented;
        }

        if (with_eltwise(0, pd) || with_eltwise(1, pd)) {
            with_eltwise_ = true;
            CHECK(create_and_set_op_descriptor(pd));
        }

        // Set parameter when post-op sum is specified
        if (with_sum(pd)) { post_op_sum_ = sum_scale(pd); }

        has_runtime_params_ = src_d.has_runtime_dims_or_strides()
                || dst_d.has_runtime_dims_or_strides()
                || weights_d.has_runtime_dims_or_strides();

        if (!has_runtime_params_) {
            // Initialise all gemm parameters if there are no runtime parameters
            init_parameters(src_d, weights_d, dst_d,
                    memory_desc_wrapper(pd->weights_md(1)));
        }

        return status::success;
    }

    bool with_bias() { return with_bias_; }
    bool has_runtime_params() { return has_runtime_params_; }

    void convert_dims_matmul(
            const memory_desc_t * desc, int *new_dims, int n_dims, bool trans=false) {
        int dims = desc->ndims;
        for (int i = 0; i < n_dims; i++) {
            new_dims[i] = 1;
        }
        for (int i = 0; i < dims - 2; i++) {
            new_dims[i + n_dims - dims] = desc->dims[i];
        }
        new_dims[n_dims - 2] = trans ? desc->dims[dims - 1] : desc->dims[dims - 2];
        new_dims[n_dims - 1] = trans ? desc->dims[dims - 2] : desc->dims[dims - 1];
    }

    status_t init_gemm_parameters(const memory_desc_wrapper src_d,
            const memory_desc_wrapper weights_d,
            const memory_desc_wrapper dst_d,
            const memory_desc_wrapper bias_d) {
        const int ndims = dst_d.ndims();
        const auto &dst_strides = &dst_d.blocking_desc().strides[ndims - 2];
        const auto &src_strides = &src_d.blocking_desc().strides[ndims - 2];
        const auto &weights_strides
                = &weights_d.blocking_desc().strides[ndims - 2];

        // A matrix is the src
        transA_ = src_strides[1] == 1 && src_d.dims()[ndims - 1] > 1
                ? false : true;
        // B matrix is the weights
        transB_ = weights_strides[1] == 1
                        && weights_d.dims()[ndims - 1] > 1
                ? false : true;
        // C matrix is the dst
        transC_ = dst_strides[1] == 1 && dst_d.dims()[ndims - 1] > 1
                ? false : true;

        return status::success;
    }

    status_t init_parameters(const memory_desc_wrapper src_d,
            const memory_desc_wrapper weights_d,
            const memory_desc_wrapper dst_d, const memory_desc_wrapper bias_d) {
        // Matmul supports runtime paramters for dimensions and scales.
        // We need to initialize them in the execute function.
        CHECK(init_gemm_parameters(src_d, weights_d, dst_d, bias_d));
        
        // Initialise cnnl tensor descriptors
        int ndims = dst_d.ndims() < 3 ? 3 : dst_d.ndims();
        int dims[NUM_IO][DNNL_MAX_NDIMS];

        convert_dims_matmul(src_d.md_, dims[src], ndims, transA_);
        convert_dims_matmul(weights_d.md_, dims[weight], ndims, transB_);
        convert_dims_matmul(dst_d.md_, dims[dst], ndims, transC_);
        CHECK(convert_data_type(src_d.md_, &data_types_[src], false));
        CHECK(convert_data_type(weights_d.md_, &data_types_[weight], false));
        CHECK(convert_data_type(dst_d.md_, &data_types_[dst], false));
        CHECK(create_and_set_tensor_descriptor_ex(&tensor_descs_[src],
                cnnlTensorLayout_t::CNNL_LAYOUT_ARRAY, data_types_[src], ndims, dims[src]));
        CHECK(create_and_set_tensor_descriptor_ex(&tensor_descs_[weight],
                cnnlTensorLayout_t::CNNL_LAYOUT_ARRAY, data_types_[weight], ndims, dims[weight]));
        CHECK(create_and_set_tensor_descriptor_ex(&tensor_descs_[dst],
                cnnlTensorLayout_t::CNNL_LAYOUT_ARRAY, data_types_[dst], ndims, dims[dst]));

        if (with_bias_) {
            // Create bias tensor descriptor
            convert_dims_matmul(bias_d.md_, dims[bias], bias_d.ndims());
            CHECK(convert_data_type(bias_d.md_, &data_types_[bias], false));
            CHECK(create_and_set_tensor_descriptor_ex(&tensor_descs_[bias],
                    cnnlTensorLayout_t::CNNL_LAYOUT_ARRAY, data_types_[bias], ndims, dims[bias]));
        }

        CNNL_EXECUTE_FUNC_V(cnnlMatMulDescCreate, &desc_);
        CNNL_EXECUTE_FUNC_V(cnnlSetMatMulDescAttr, desc_,
                cnnlMatMulDescAttribute_t::CNNL_MATMUL_DESC_COMPUTE_TYPE,
                &acc_type_, sizeof(acc_type_));
        int transA = transA_ ^ transC_, transB = transB_ ^ transC_;
        CNNL_EXECUTE_FUNC_V(cnnlSetMatMulDescAttr, desc_,
                cnnlMatMulDescAttribute_t::CNNL_MATMUL_DESC_TRANSA, &transA,
                sizeof(transA));
        CNNL_EXECUTE_FUNC_V(cnnlSetMatMulDescAttr, desc_,
                cnnlMatMulDescAttribute_t::CNNL_MATMUL_DESC_TRANSB, &transB,
                sizeof(transB));

        return status::success;
    }

    void execute(cnnlHandle_t handle,
            void *a_, void *b_, void *c, void *bias,
            void *src_scale, void *wei_scale, void *dst_scale) {
        assert(c != bias);
        cnrtQueue_t queue;
        CNRT_CHECK(cnrtQueueCreate(&queue));
        CNNL_EXECUTE_FUNC_V(cnnlSetQueue, handle, queue);
        auto &a_desc = transC_ ? tensor_descs_[weight] : tensor_descs_[src];
        auto &b_desc = transC_ ? tensor_descs_[src] : tensor_descs_[weight];
        auto &c_desc = tensor_descs_[dst];
        auto a = transC_ ? b_ : a_;
        auto b = transC_ ? a_ : b_;
        if (algo_ == nullptr)
            CNNL_EXECUTE_FUNC_V(cnnlMatMulAlgoCreate, &algo_);
        if (heuristic_result_ == nullptr)
            CNNL_EXECUTE_FUNC_V(cnnlCreateMatMulHeuristicResult, &heuristic_result_);
        int requested_algo_count = 1, return_algo_count = 0;
        CNNL_EXECUTE_FUNC_V(cnnlGetMatMulAlgoHeuristic, handle, desc_, a_desc, b_desc, c_desc, c_desc,
                                    nullptr /* prefer */, requested_algo_count, &heuristic_result_,
                                    &return_algo_count);
        CNNL_EXECUTE_FUNC_V(cnnlGetMatMulHeuristicResult, heuristic_result_, algo_, &workspace_size_);
        if (workspace_size_ > 0) {
            BANG_EXECUTE_FUNC_V(cnMalloc, (CNaddr *)&workspace_, workspace_size_);
        }

        float gemm_beta = post_op_sum_;
        float scale = 1.0f;
        float host_dst_scale = 1.0f;
        if (src_scale) {
            float host_src_scale = 1.0f;
            BANG_EXECUTE_FUNC_S(cnMemcpy, (CNaddr)&host_src_scale,
                    (CNaddr)src_scale, sizeof(float));
            scale *= host_src_scale;
        }
        if (wei_scale) {
            float host_wei_scale = 1.0f;
            BANG_EXECUTE_FUNC_S(cnMemcpy, (CNaddr)&host_wei_scale,
                    (CNaddr)wei_scale, sizeof(float));
            scale *= host_wei_scale;
        }
        if (dst_scale) {
            BANG_EXECUTE_FUNC_S(cnMemcpy, (CNaddr)&host_dst_scale,
                    (CNaddr)dst_scale, sizeof(float));
            // For eltwise post-ops, apply the dst scale afterward
            if (!with_eltwise_) scale /= host_dst_scale;
        }

        CNNL_EXECUTE_FUNC(cnnlBatchMatMulBCast_v2, handle, desc_, algo_,
                &scale, a_desc, a, b_desc, b, &gemm_beta, c_desc, c,
                workspace_, workspace_size_);
            
        if (with_bias_) {
            // When bias is specified call cnnlAssignAdd()
            float bias_beta = 1;
            scale = (with_eltwise_ ? 1 : 1.0f / host_dst_scale);
            CNNL_EXECUTE_FUNC_V(cnnlGetAssignAddWorkspaceSize, handle, tensor_descs_[io::bias], c_desc, &workspace_add_size_);
            if (workspace_add_size_ > 0) {
                BANG_EXECUTE_FUNC_V(cnMalloc, (CNaddr *)&workspace_add_, workspace_add_size_);
            }
            CNNL_EXECUTE_FUNC_V(cnnlAssignAdd, handle, &scale, tensor_descs_[io::bias], bias, workspace_add_, workspace_add_size_, &bias_beta, c_desc, c);
        }

        if (with_eltwise_) {
            // Perform elementwise operation if specified
            float alpha = 1.0f / host_dst_scale;
            float beta = 0;
            CNNL_EXECUTE_FUNC(cnnlActivationForward, handle, act_desc_,
                    &alpha, c_desc, c, &beta, c_desc, c);
        }
        CNRT_CHECK(cnrtQueueSync(queue));

        if (workspace_ != nullptr) {
            BANG_EXECUTE_FUNC_V(cnFree, (CNaddr)workspace_);
            workspace_ = nullptr;
        }
        if (workspace_add_ != nullptr) {
            BANG_EXECUTE_FUNC_V(cnFree, (CNaddr)workspace_add_);
            workspace_add_ = nullptr;
        }
        CNRT_CHECK(cnrtQueueDestroy(queue));
    }

    ~cnnl_matmul_impl_t() { cleanup(); }

    void cleanup() {
        if (act_desc_) {
            CNNL_EXECUTE_FUNC_V(cnnlDestroyActivationDescriptor, act_desc_);
            act_desc_ = nullptr;
        }
        for (size_t i = 0; i < NUM_IO; i++) {
            if (tensor_descs_[i]) {
                CNNL_EXECUTE_FUNC_V(
                        cnnlDestroyTensorDescriptor, tensor_descs_[i]);
                tensor_descs_[i] = nullptr;
            }
        }
        if (desc_)
            CNNL_EXECUTE_FUNC_V(cnnlMatMulDescDestroy, desc_);
        if (algo_)
            CNNL_EXECUTE_FUNC_V(cnnlMatMulAlgoDestroy, algo_);
        if (heuristic_result_)
            CNNL_EXECUTE_FUNC_V(
                    cnnlDestroyMatMulHeuristicResult, heuristic_result_);
    }

private:
    bool transA_;
    bool transB_;
    bool transC_;
    bool isbatched_ = false, with_bias_ = false;
    bool with_eltwise_ = false;
    bool has_runtime_params_ = false;
    enum io { src = 0, weight, dst, bias, NUM_IO };
    cnnlDataType_t data_types_[NUM_IO];
    cnnlDataType_t acc_type_ = cnnlDataType_t::CNNL_DTYPE_FLOAT;
    cnnlTensorDescriptor_t tensor_descs_[NUM_IO];
    cnnlMatMulDescriptor_t desc_ = nullptr;
    cnnlMatMulAlgo_t algo_ = nullptr;
    cnnlMatMulHeuristicResult_t heuristic_result_ = nullptr;
    void *workspace_ = nullptr, *workspace_add_ = nullptr;
    size_t workspace_size_ = 0, workspace_add_size_ = 0;
    cnnlActivationDescriptor_t act_desc_ = nullptr;
    float post_op_sum_ = 0.0f;
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
