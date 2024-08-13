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

struct cnnl_reorder_impl_t {
public:
    status_t init(const reorder_pd_t *pd) {
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
        beta_ = pd->beta();

        CHECK(convert_data_type(pd->src_md(), &src_data_type_));
        CHECK(convert_data_type(pd->dst_md(), &dst_data_type_));

        convert_dims(pd->src_md()->padded_dims, src_dims_, pd->src_md()->ndims);
        convert_dims(pd->dst_md()->padded_dims, dst_dims_, pd->dst_md()->ndims);
        convert_dims(pd->src_md()->format_desc.blocking.strides, src_strides_,
                pd->src_md()->ndims);
        convert_dims(pd->dst_md()->format_desc.blocking.strides, dst_strides_,
                pd->dst_md()->ndims);

        ndims_ = pd->dst_md()->ndims > 4 ? pd->dst_md()->ndims : 4;
        src_size_ = memory_desc_wrapper(pd->src_md()).size();
        dst_size_ = memory_desc_wrapper(pd->dst_md()).size();

        if (src_data_type_ == dst_data_type_) {
            // when dst and src data types match, call cnnlTranspose
            get_format(pd->src_md(), src_format_);
            get_format(pd->dst_md(), dst_format_);
            if (!get_permute(dst_format_, src_format_)) return status::unimplemented;
            CHECK(transpose_dims(src_dims_, ndims_, src_format_));
            CHECK(transpose_dims(dst_dims_, ndims_, dst_format_));
            CHECK(CNNL_EXECUTE_FUNC_S(
                    cnnlCreateTransposeDescriptor, &trans_desc_));
            CHECK(CNNL_EXECUTE_FUNC_S(cnnlSetTransposeDescriptor, trans_desc_,
                    ndims_, permute_.data()));
            CHECK(create_and_set_tensor_descriptor_ex(&src_desc_, src_format_, src_data_type_, 
                    ndims_, src_dims_));
            CHECK(create_and_set_tensor_descriptor_ex(&dst_desc_, dst_format_, dst_data_type_,
                    ndims_, dst_dims_));
        } else {
            // otherwise, call cnnlCastDataType for typecast and transpose
            if (!get_cast_type(dst_data_type_, src_data_type_)) return status::unimplemented;
            CHECK(create_and_set_tensor_descriptor(&src_desc_, src_data_type_, ndims_, src_dims_,
                    src_strides_));
            CHECK(create_and_set_tensor_descriptor(&dst_desc_, dst_data_type_, ndims_, dst_dims_,
                    dst_strides_));
        }

        return status::success;
    }

    bool get_permute(cnnlTensorLayout_t dst_format, cnnlTensorLayout_t src_format) {
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

    bool get_cast_type(cnnlDataType_t dst_data_type, cnnlDataType_t src_data_type) {
        do_rounding_ = 
            (src_data_type_ == cnnlDataType_t::CNNL_DTYPE_FLOAT || src_data_type_ == cnnlDataType_t::CNNL_DTYPE_HALF) && 
            (dst_data_type_ == cnnlDataType_t::CNNL_DTYPE_INT32 || dst_data_type_ == cnnlDataType_t::CNNL_DTYPE_INT8 || 
                dst_data_type_ == cnnlDataType_t::CNNL_DTYPE_UINT8 || dst_data_type_ == cnnlDataType_t::CNNL_DTYPE_BOOL);
        switch (src_data_type_) {
            case cnnlDataType_t::CNNL_DTYPE_FLOAT:
                switch (dst_data_type_) {
                    case cnnlDataType_t::CNNL_DTYPE_HALF:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_FLOAT_TO_HALF;
                        break;
                    case cnnlDataType_t::CNNL_DTYPE_BOOL:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_FLOAT_TO_BOOL;
                        break;
                    case cnnlDataType_t::CNNL_DTYPE_INT8:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_FLOAT_TO_INT8;
                        break;
                    case cnnlDataType_t::CNNL_DTYPE_UINT8:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_FLOAT_TO_UINT8;
                        break;
                    case cnnlDataType_t::CNNL_DTYPE_INT32:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_FLOAT_TO_INT32;
                        break;
                    default:
                        return false;
                }
                break;
            case cnnlDataType_t::CNNL_DTYPE_HALF:
                switch (dst_data_type_) {
                    case cnnlDataType_t::CNNL_DTYPE_FLOAT:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_HALF_TO_FLOAT;
                        break;
                    case cnnlDataType_t::CNNL_DTYPE_BOOL:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_HALF_TO_BOOL;
                        break;
                    case cnnlDataType_t::CNNL_DTYPE_INT8:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_HALF_TO_INT8;
                        break;
                    case cnnlDataType_t::CNNL_DTYPE_UINT8:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_HALF_TO_UINT8;
                        break;
                    case cnnlDataType_t::CNNL_DTYPE_INT32:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_HALF_TO_INT32;
                        break;
                    default:
                        return false;
                }
                break;
            case cnnlDataType_t::CNNL_DTYPE_BOOL:
                switch (dst_data_type_) {
                    case cnnlDataType_t::CNNL_DTYPE_FLOAT:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_BOOL_TO_FLOAT;
                        break;
                    case cnnlDataType_t::CNNL_DTYPE_HALF:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_BOOL_TO_HALF;
                        break;
                    case cnnlDataType_t::CNNL_DTYPE_INT32:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_BOOL_TO_INT32;
                        break;
                    default:
                        return false;
                }
                break;
            case cnnlDataType_t::CNNL_DTYPE_INT8:
                switch (dst_data_type_) {
                    case cnnlDataType_t::CNNL_DTYPE_FLOAT:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_INT8_TO_FLOAT;
                        break;
                    case cnnlDataType_t::CNNL_DTYPE_HALF:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_INT8_TO_HALF;
                        break;
                    case cnnlDataType_t::CNNL_DTYPE_INT32:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_INT8_TO_INT32;
                        break;
                    default:
                        return false;
                }
                break;
            case cnnlDataType_t::CNNL_DTYPE_UINT8:
                switch (dst_data_type_) {
                    case cnnlDataType_t::CNNL_DTYPE_FLOAT:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_UINT8_TO_FLOAT;
                        break;
                    case cnnlDataType_t::CNNL_DTYPE_HALF:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_UINT8_TO_HALF;
                        break;
                    case cnnlDataType_t::CNNL_DTYPE_INT32:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_UINT8_TO_INT32;
                        break;
                    default:
                        return false;
                }
                break;
            case cnnlDataType_t::CNNL_DTYPE_INT32:
                switch (dst_data_type_) {
                    case cnnlDataType_t::CNNL_DTYPE_FLOAT:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_INT32_TO_FLOAT;
                        break;
                    case cnnlDataType_t::CNNL_DTYPE_HALF:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_INT32_TO_HALF;
                        break;
                    case cnnlDataType_t::CNNL_DTYPE_BOOL:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_INT32_TO_BOOL;
                        break;
                    case cnnlDataType_t::CNNL_DTYPE_INT8:
                        cast_type_ = cnnlCastDataType_t::CNNL_CAST_INT32_TO_INT8;
                        break;
                    default:
                        return false;
                }
                break;
            default: return false;
        }
        return true;
    }

    void execute(cnnlHandle_t handle, void *src, void *dst, void *src_scale, void *dst_scale) const {
        float alpha = 1.0f;
        if (src_scale) {
            float host_src_scale = 1.0f;
            BANG_EXECUTE_FUNC(cnMemcpy, (CNaddr)&host_src_scale,
                    (CNaddr)src_scale, sizeof(float));
            alpha *= host_src_scale;
        }
        float beta = beta_;
        if (dst_scale) {
            float host_dst_scale = 1.0f;
            BANG_EXECUTE_FUNC(cnMemcpy, (CNaddr)&host_dst_scale,
                    (CNaddr)dst_scale, sizeof(float));
            alpha /= host_dst_scale;
            beta /= host_dst_scale;
        }
        bool do_scaling = (alpha != 1.0f || beta != 0.0f);
        void *reorder_dst = dst;
        if (do_scaling) {
            if (scratchpad_dst_ == nullptr)
                BANG_EXECUTE_FUNC(cnMalloc, (CNaddr *)&scratchpad_dst_, dst_size_);
            reorder_dst = scratchpad_dst_;
        }

        if (src_data_type_ == dst_data_type_) {
            CNNL_EXECUTE_FUNC(cnnlTranspose, handle, trans_desc_, 
                    src_desc_, src, dst_desc_, reorder_dst);
        } else {
            if (do_rounding_) {
                if (scratchpad_src_ == nullptr)
                    BANG_EXECUTE_FUNC(cnMalloc, (CNaddr *)&scratchpad_src_, src_size_);
                CNNL_EXECUTE_FUNC(cnnlRound, handle, src_desc_, src, src_desc_, scratchpad_src_);
                src = scratchpad_src_;
            }
            CNNL_EXECUTE_FUNC(cnnlCastDataType, handle, src_desc_, 
                    src, cast_type_, dst_desc_, reorder_dst);
        }
        if (do_scaling) {
            CNNL_EXECUTE_FUNC(cnnlCaxpby, handle, &beta, dst_desc_, dst, &alpha, dst_desc_, reorder_dst);
        }
    }

    ~cnnl_reorder_impl_t() {
        if (src_desc_)
            CNNL_EXECUTE_FUNC_V(cnnlDestroyTensorDescriptor, src_desc_);
        if (dst_desc_)
            CNNL_EXECUTE_FUNC_V(cnnlDestroyTensorDescriptor, dst_desc_);
        if (trans_desc_)
            CNNL_EXECUTE_FUNC_V(cnnlDestroyTransposeDescriptor, trans_desc_);
        if (scratchpad_src_)
            BANG_EXECUTE_FUNC(cnFree, (CNaddr)scratchpad_src_);
        if (scratchpad_dst_)
            BANG_EXECUTE_FUNC(cnFree, (CNaddr)scratchpad_dst_);
    }

    int dst_offset_in_bytes() { return dst_offset_in_bytes_; }
    int src_offset_in_bytes() { return src_offset_in_bytes_; }

protected:
    cnnlDataType_t src_data_type_;
    cnnlDataType_t dst_data_type_;
    int ndims_;
    int src_dims_[DNNL_MAX_NDIMS];
    int dst_dims_[DNNL_MAX_NDIMS];
    int src_strides_[DNNL_MAX_NDIMS];
    int dst_strides_[DNNL_MAX_NDIMS];
    cnnlTensorDescriptor_t src_desc_;
    cnnlTensorDescriptor_t dst_desc_;
    size_t src_size_;
    size_t dst_size_;
    // used for rounding src data before typecast
    mutable void *scratchpad_src_ = nullptr;
    // store the result of reorder for scaling
    mutable void *scratchpad_dst_ = nullptr;
    cnnlTransposeDescriptor_t trans_desc_;
    cnnlCastDataType_t cast_type_;
    bool do_rounding_ = false;
    cnnlTensorLayout_t src_format_;
    cnnlTensorLayout_t dst_format_;
    std::vector<int> permute_;
    float beta_ = 0.0f;
    int dst_offset_in_bytes_ = 0;
    int src_offset_in_bytes_ = 0;
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
