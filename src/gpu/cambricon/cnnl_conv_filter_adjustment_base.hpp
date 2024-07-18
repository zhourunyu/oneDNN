/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef GPU_CAMBRICON_CNNL_CONV_FILTER_ADJUSTMENT_BASE_HPP
#define GPU_CAMBRICON_CNNL_CONV_FILTER_ADJUSTMENT_BASE_HPP

#include "cnnl.h"

#include "common/type_helpers.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/cambricon/sycl_bang_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct cnnl_conv_filter_adjustment_base_t {
public:
    cnnlTensorDescriptor_t current_filter_desc_, transform_filter_desc_;
    cnnlTransposeDescriptor_t trans_desc_, trans_desc_reverse_;
    std::vector<int> permute_, permute_reverse_;
    // for filter in convolution, cnnl only support nhwc and hwcn.
    // the hwio and dhwio is not supported and should be converted
    // to either of the above format.
    virtual bool supported_filter_format(const memory_desc_t *md) const {
        const memory_desc_wrapper mem_wrapper(md);
        /// NOTE: the transformation for oidhw to oihwd is disabled until cNNL
        // fixes the the current bug for oihwd format. the transformation for
        // odhwi to ohwdi has been disabled until cNNL provides support for
        // 3d convolution in ohwdi format.
        return (!(mem_wrapper.matches_one_of_tag(/*format_tag::oidhw,*/
                /*format_tag::odhwi,*/ format_tag::dhwio, format_tag::hwio)));
    }

    virtual ~cnnl_conv_filter_adjustment_base_t() {
        if (current_filter_desc_) {
            CNNL_EXECUTE_FUNC_V(
                    cnnlDestroyTensorDescriptor, current_filter_desc_);
        }
        if (transform_filter_desc_) {
            CNNL_EXECUTE_FUNC_V(
                    cnnlDestroyTensorDescriptor, transform_filter_desc_);
        }
        if (trans_desc_) {
            CNNL_EXECUTE_FUNC_V(
                    cnnlDestroyTransposeDescriptor, trans_desc_);
        }
        if (trans_desc_reverse_) {
            CNNL_EXECUTE_FUNC_V(
                    cnnlDestroyTransposeDescriptor, trans_desc_reverse_);
        }
    }

    void set_permute_dims(int ndims, cnnlTensorLayout_t from_format,
            cnnlTensorLayout_t to_format) {
        if (from_format == CNNL_LAYOUT_NCHW && to_format == CNNL_LAYOUT_NHWC) {
            switch (ndims) {
                case 4:
                    permute_ = {0, 2, 3, 1};
                    permute_reverse_ = {0, 3, 1, 2};
                    break;
                case 5:
                    permute_ = {0, 1, 3, 4, 2};
                    permute_reverse_ = {0, 1, 4, 2, 3};
                    break;
                default:
                    assert(!"Unsupported ndims");
            }
        } else if (from_format == CNNL_LAYOUT_NCHW && to_format == CNNL_LAYOUT_HWCN) {
            switch (ndims) {
                case 4:
                    permute_ = {2, 3, 1, 0};
                    permute_reverse_ = {3, 2, 0, 1};
                    break;
                case 5:
                    permute_ = {3, 4, 2, 0, 1};
                    permute_reverse_ = {3, 4, 2, 0, 1};
                    break;
                default:
                    assert(!"Unsupported ndims");
            
            }
        } else
            assert(!"Unsupported layout");
    }

    virtual status_t init_filter_transformation(
            cnnlDataType_t filter_data_types, int filter_ndims,
            int *filter_dims, cnnlTensorLayout_t filter_format,
            cnnlTensorLayout_t transform_filter_format) {
        // Set a descriptor for the current filter.
        CHECK(create_and_set_tensor_descriptor_ex(&current_filter_desc_,
                filter_format, filter_data_types, filter_ndims, filter_dims));
        // Set a descriptor for the transform filter.
        CHECK(create_and_set_tensor_descriptor_ex(&transform_filter_desc_,
                transform_filter_format, filter_data_types, filter_ndims, 
                filter_dims));
        set_permute_dims(filter_ndims, filter_format, transform_filter_format);
        CNNL_EXECUTE_FUNC_V(cnnlCreateTransposeDescriptor, &trans_desc_);
        CNNL_EXECUTE_FUNC_V(cnnlCreateTransposeDescriptor, &trans_desc_reverse_);
        CNNL_EXECUTE_FUNC_V(cnnlSetTransposeDescriptor, trans_desc_, filter_ndims, 
                permute_.data());
        CNNL_EXECUTE_FUNC_V(cnnlSetTransposeDescriptor, trans_desc_reverse_, 
                filter_ndims, permute_reverse_.data());
        return status::success;
    }

    void transform_filter(cnnlHandle_t handle, void *current_filter,
            void *transform_filter) const {
        CNNL_EXECUTE_FUNC(cnnlTranspose, handle, trans_desc_,
                current_filter_desc_, current_filter, transform_filter_desc_, 
                transform_filter);
    }
    void undo_transform_filter(cnnlHandle_t handle, void *transform_filter,
            void *current_filter) const {
        CNNL_EXECUTE_FUNC(cnnlTranspose, handle, trans_desc_reverse_,
                transform_filter_desc_, transform_filter, 
                current_filter_desc_, current_filter);
    }
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
