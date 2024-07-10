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

#ifndef GPU_CAMBRICON_CNNL_REORDER_IMPL_HPP
#define GPU_CAMBRICON_CNNL_REORDER_IMPL_HPP

#include "common/type_helpers.hpp"
#include "gpu/cambricon/sycl_bang_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct cnnl_reorder_generic_t {
public:
    virtual status_t init(const reorder_pd_t *pd) = 0;

    virtual void execute(cnnlHandle_t handle, void *src, void *dst) const = 0;

    virtual ~cnnl_reorder_generic_t() {
        CNNL_EXECUTE_FUNC_V(cnnlDestroyTensorDescriptor, src_desc_);
        CNNL_EXECUTE_FUNC_V(cnnlDestroyTensorDescriptor, dst_desc_);
        CNNL_EXECUTE_FUNC_V(cnnlDestroyTransposeDescriptor, trans_desc_);
    }

    int dst_offset_in_bytes() { return dst_offset_in_bytes_; }
    int src_offset_in_bytes() { return src_offset_in_bytes_; }

protected:
    cnnlDataType_t src_data_type_;
    cnnlDataType_t dst_data_type_;
    int ndims_;
    int dims_[DNNL_MAX_NDIMS];
    int dst_dims_[DNNL_MAX_NDIMS];
    cnnlTensorDescriptor_t src_desc_;
    cnnlTensorDescriptor_t dst_desc_;
    cnnlTransposeDescriptor_t trans_desc_;
    int dst_offset_in_bytes_ = 0;
    int src_offset_in_bytes_ = 0;
};

// This structure is used when the memory format includes blocking
struct cnnl_reorder_stride_t : public cnnl_reorder_generic_t {
public:
    status_t init(const reorder_pd_t *pd) override {
        // If any of the dimensions are 0 we should not continue with creating
        // cnnl descriptors
        memory_desc_wrapper wrap(pd->src_md());
        if (wrap.size() == 0) { return status::success; }
        // Validity checks
        assert(pd->dst_md()->ndims == pd->src_md()->ndims);

        dst_offset_in_bytes_ = pd->dst_md()->offset0
                * types::data_type_size(pd->dst_md()->data_type);
        src_offset_in_bytes_ = pd->src_md()->offset0
                * types::data_type_size(pd->src_md()->data_type);

        CHECK(convert_data_type(pd->src_md(), &src_data_type_));
        CHECK(convert_data_type(pd->dst_md(), &dst_data_type_));

        convert_dims(pd->src_md()->padded_dims, dims_, pd->src_md()->ndims);
        convert_dims(pd->dst_md()->padded_dims, dst_dims_, pd->dst_md()->ndims);

        ndims_ = pd->dst_md()->ndims > 4 ? pd->dst_md()->ndims : 4;

        bool ok = layout_n_permute(pd);
        if(!ok) return status::invalid_arguments;

        CHECK(transpose_dims(dims_, ndims_, src_format_));
        CHECK(transpose_dims(dst_dims_, ndims_, dst_format_));

        // Create and set tensor transform descriptor
        CHECK(CNNL_EXECUTE_FUNC_S(
                cnnlCreateTransposeDescriptor, &trans_desc_));
        CHECK(CNNL_EXECUTE_FUNC_S(cnnlSetTransposeDescriptor, trans_desc_,
                ndims_, permute_.data()));
        // Create and set source tensor descriptor
        CHECK(CNNL_EXECUTE_FUNC_S(cnnlCreateTensorDescriptor, &src_desc_));
        CHECK(CNNL_EXECUTE_FUNC_S(cnnlSetTensorDescriptor, src_desc_, src_format_, src_data_type_, ndims_, dims_));
        // Create and set destination tensor descriptor
        CHECK(CNNL_EXECUTE_FUNC_S(cnnlCreateTensorDescriptor, &dst_desc_));
        CHECK(CNNL_EXECUTE_FUNC_S(cnnlSetTensorDescriptor, dst_desc_, dst_format_, dst_data_type_, ndims_, dst_dims_));

        return status::success;
    }

    bool layout_n_permute(const reorder_pd_t *pd) {

        get_format(pd->src_md(), src_format_);
        get_format(pd->dst_md(), dst_format_);

        if (src_format_ == cnnlTensorLayout_t::CNNL_LAYOUT_NCHW) {
            if (dst_format_ == cnnlTensorLayout_t::CNNL_LAYOUT_NCHW) 
                permute_ = {0, 1, 2, 3};
            else if (dst_format_ == cnnlTensorLayout_t::CNNL_LAYOUT_NHWC)
                permute_ = {0, 2, 3, 1};
            else
                return false;
        } else if (src_format_ == cnnlTensorLayout_t::CNNL_LAYOUT_NHWC) {
            if (dst_format_ == cnnlTensorLayout_t::CNNL_LAYOUT_NHWC) 
                permute_ = {0, 1, 2, 3};
            else if (dst_format_ == cnnlTensorLayout_t::CNNL_LAYOUT_NCHW)
                permute_ = {0, 3, 1, 2};
            else
                return false;
        } else
            return false;
        return true;
    }

    void execute(cnnlHandle_t handle, void *src, void *dst) const override {
        // cnnlTranspose() function is required to support blocking.
        // It requires the output tensor to be in cnnl supported format.

        CNNL_EXECUTE_FUNC(cnnlTranspose, handle, trans_desc_, 
                src_desc_, src, dst_desc_, dst);
    }

private:
    cnnlTensorLayout_t src_format_;
    cnnlTensorLayout_t dst_format_;
    std::vector<int> permute_;

    using cnnl_reorder_generic_t::cnnl_reorder_generic_t;
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
