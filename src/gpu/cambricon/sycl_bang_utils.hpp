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

#ifndef GPU_CAMBRICON_SYCL_BANG_UTILS_HPP
#define GPU_CAMBRICON_SYCL_BANG_UTILS_HPP

#include <cn_api.h>
#include <cnnl.h>
#include <stdexcept>

#include "dnnl_sycl.hpp"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/primitive_attr.hpp"
#include "common/z_magic.hpp"

#include "sycl/sycl_utils.hpp"

#include "gpu/cambricon/sycl_bang_compat.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

#define CTX_OUT_ACCESSOR(arg) \
    utils::downcast<sycl::sycl_buffer_memory_storage_t *>( \
            &CTX_OUT_STORAGE(arg)) \
            ->buffer() \
            .get_access<::sycl::access::mode::write>(cgh)

#define CTX_IN_ACCESSOR(arg) \
    utils::downcast<sycl::sycl_buffer_memory_storage_t *>( \
            &CTX_IN_STORAGE(arg)) \
            ->buffer() \
            .get_access<::sycl::access::mode::read>(cgh)

#define CTX_SCRATCH_ACCESSOR(arg) \
    utils::downcast<sycl::sycl_buffer_memory_storage_t *>( \
            ctx.get_scratchpad_grantor().get_memory_storage(arg).get()) \
            ->buffer() \
            .get_access<::sycl::access::mode::read_write>(cgh)

bool compare_bang_devices(const ::sycl::device &lhs, const ::sycl::device &rhs);
bool has_bf16_support(const ::sycl::device &dev);

// Check if the device type matches the passed engine kind
inline status_t check_device(dnnl::impl::engine_kind_t eng_kind) {
    return (eng_kind == dnnl::impl::engine_kind::gpu
                    ? status::success
                    : status::invalid_arguments);
}

static void convert_dnnl_dims_array(
        const dnnl_dim_t *dims, int *new_dims, int n_dims) {
    for (size_t i = 0; i < n_dims; i++) {
        new_dims[i] = static_cast<int>(dims[i]);
    }
}

static void convert_dims(const dnnl_dim_t *dims, int *new_dims, int n_dims,
        int adjustment_size = 4, int adjustment_value = 1) {
    convert_dnnl_dims_array(dims, new_dims, n_dims);
    for (size_t i = n_dims; i < adjustment_size; i++) {
        new_dims[i] = adjustment_value;
    }
}

// transpose dims from nchw to actual layout
static status_t transpose_dims(int *dims, int ndims, cnnlTensorLayout_t layout) {
    switch (layout) {
        case cnnlTensorLayout_t::CNNL_LAYOUT_NCHW:
            return status::success;
        case cnnlTensorLayout_t::CNNL_LAYOUT_NHWC:
            assert(ndims >= 3);
            std::swap(dims[ndims - 3], dims[ndims - 2]);
            std::swap(dims[ndims - 2], dims[ndims - 1]);
            return status::success;
        case cnnlTensorLayout_t::CNNL_LAYOUT_HWCN:
            assert(ndims >= 4);
            std::swap(dims[ndims - 4], dims[ndims - 2]);
            std::swap(dims[ndims - 3], dims[ndims - 1]);
            std::swap(dims[ndims - 2], dims[ndims - 1]);
            return status::success;
        default: return status::unimplemented;
    }
}

static bool memory_desc_matches_nchw_vect_c(const memory_desc_t *mem_desc) {
    // Only one block is supported for second (C) dimension and the block size
    // must be 4 and the dimension has to be a multiple of block size.
    auto is_int_8 = utils::one_of(mem_desc->data_type, data_type::s8);
    auto &strides = mem_desc->format_desc.blocking.strides;
    if (is_int_8 && mem_desc->format_desc.blocking.inner_nblks == 1
            && mem_desc->format_desc.blocking.inner_idxs[0] == 1
            && mem_desc->format_desc.blocking.inner_blks[0] == 4
            && mem_desc->dims[1] % 4 == 0) {
        for (int d = 0; d < mem_desc->ndims - 1; ++d)
            if (strides[d] < strides[d + 1]) return false;
        return true;
    }
    return false;
}

static bool has_different_block_size(
        const memory_desc_t *src_md, const memory_desc_t *dst_md) {
    return ((src_md->format_desc.blocking.inner_nblks > 0
                    && dst_md->format_desc.blocking.inner_nblks == 0)
            || (src_md->format_desc.blocking.inner_nblks == 0
                    && dst_md->format_desc.blocking.inner_nblks > 0));
}
static bool adjust_dim_for_dnn(
        int *dims, int n_dims, const memory_desc_t *mem_desc) {
    if (memory_desc_matches_nchw_vect_c(mem_desc)) {
        dims[n_dims] = mem_desc->format_desc.blocking.inner_blks[0];
        dims[mem_desc->format_desc.blocking.inner_idxs[0]]
                /= mem_desc->format_desc.blocking.inner_blks[0];
        return true;
    }
    return false;
}

static bool adjust_stride_for_dnn(
        int *stride, int n_dims, const memory_desc_t *mem_desc) {
    if (memory_desc_matches_nchw_vect_c(mem_desc)) {
        stride[n_dims] = mem_desc->format_desc.blocking.inner_nblks;
        return true;
    }
    return false;
}

// Check if the dimensions contain any zeros, returns true if they do.
static bool has_zero_dims(const dnnl_dim_t *dims, int n_dims) {
    for (size_t i = 0; i < n_dims; i++) {
        if (dims[i] == 0) { return true; }
    }
    return false;
}

static status_t get_format(const memory_desc_t *md, cnnlTensorLayout_t &format,
        bool consider_ab_as_nhwc = false) {
    const memory_desc_wrapper mem_wrapper(md);
    if (mem_wrapper.matches_one_of_tag(format_tag::ab, format_tag::abc,
                       format_tag::abcd, format_tag::abcde,
                       format_tag::abcdef)) {
        format = cnnlTensorLayout_t::CNNL_LAYOUT_NCHW;
    } else if (mem_wrapper.matches_one_of_tag(
                       format_tag::acb, format_tag::acdb, format_tag::acdeb)) {
        format = cnnlTensorLayout_t::CNNL_LAYOUT_NHWC;
    } else if (mem_wrapper.matches_one_of_tag(format_tag::ndhwc)) {
        format = cnnlTensorLayout_t::CNNL_LAYOUT_NDHWC;
    } else if (mem_wrapper.matches_one_of_tag(format_tag::any)) {
        format = cnnlTensorLayout_t::CNNL_LAYOUT_ARRAY;
    } else {
        return status::unimplemented;
    }
    if (consider_ab_as_nhwc && mem_wrapper.matches_one_of_tag(format_tag::ab)) {
        format = cnnlTensorLayout_t::CNNL_LAYOUT_NHWC;
    }
    return status::success;
}

static bool memory_format_ok(const memory_desc_t *mem_desc) {
    return (memory_desc_matches_nchw_vect_c(mem_desc)
            || mem_desc->format_desc.blocking.inner_nblks == 0);
}

static status_t convert_data_type(const memory_desc_t *mem_desc,
        cnnlDataType_t *cnnl_data_type, bool vectorized = true) {
    switch (mem_desc->data_type) {
        case dnnl_data_type_t::dnnl_f16:
            *cnnl_data_type = cnnlDataType_t::CNNL_DTYPE_HALF;
            break;
        case dnnl_data_type_t::dnnl_bf16:
            *cnnl_data_type = cnnlDataType_t::CNNL_DTYPE_BFLOAT16;
            break;
        case dnnl_data_type_t::dnnl_f32:
            *cnnl_data_type = cnnlDataType_t::CNNL_DTYPE_FLOAT;
            break;
        case dnnl_data_type_t::dnnl_s8:
            *cnnl_data_type = cnnlDataType_t::CNNL_DTYPE_INT8;
            break;
        default: return status::unimplemented;
    }
    return status::success;
}

class bang_error : virtual public std::runtime_error {

protected:
    inline const char *bang_error_map(CNresult result) {
        switch (result) {
            case CN_SUCCESS: return "CN_SUCCESS";
            case CN_OPS_ERROR_NOT_PERMITTED: return "CN_OPS_ERROR_NOT_PERMITTED";
            case CN_CONTEXT_ERROR_INVALID:
                return "CN_CONTEXT_ERROR_INVALID";
            case CN_ERROR_INVALID_DEVICE: return "CN_ERROR_INVALID_DEVICE";
            case CN_ERROR_INVALID_VALUE: return "CN_ERROR_INVALID_VALUE";
            case CN_MEMORY_ERROR_OUT_OF_MEMORY: return "CN_MEMORY_ERROR_OUT_OF_MEMORY";
            case CN_INVOKE_ERROR_OUT_OF_RESOURCES:
                return "CN_INVOKE_ERROR_OUT_OF_RESOURCES";
            default: return "<unknown>";
        }
    }
    int error_number_;

public:
    explicit bang_error(const std::string &message, CNresult result)
        : std::runtime_error((message + std::string(bang_error_map(result)))) {
        error_number_ = static_cast<int>(result);
    }
    virtual ~bang_error() throw() {}

    virtual int get_error_number() const throw() { return error_number_; }
};

class cnnl_error : virtual public std::runtime_error {

protected:
    inline const char *cnnl_get_error_string(cnnlStatus_t status) {
        switch (status) {
            case CNNL_STATUS_SUCCESS: return "CNNL_STATUS_SUCCESS";
            case CNNL_STATUS_NOT_INITIALIZED:
                return "CNNL_STATUS_NOT_INITIALIZED";
            case CNNL_STATUS_ALLOC_FAILED: return "CNNL_STATUS_ALLOC_FAILED";
            case CNNL_STATUS_BAD_PARAM: return "CNNL_STATUS_BAD_PARAM";
            case CNNL_STATUS_INTERNAL_ERROR:
                return "CNNL_STATUS_INTERNAL_ERROR";
            case CNNL_STATUS_ARCH_MISMATCH:
                return "CNNL_STATUS_ARCH_MISMATCH";
            case CNNL_STATUS_EXECUTION_FAILED:
                return "CNNL_STATUS_EXECUTION_FAILED";
            case CNNL_STATUS_NOT_SUPPORTED:
                return "CNNL_STATUS_NOT_SUPPORTED";
            case CNNL_STATUS_NUMERICAL_OVERFLOW:
                return "CNNL_STATUS_NUMERICAL_OVERFLOW";
            default: return "<unknown>";
        }
    }
    int error_number_;

public:
    explicit cnnl_error(const std::string &message, cnnlStatus_t result)
        : std::runtime_error(
                (message + std::string(cnnl_get_error_string(result)))) {
        error_number_ = static_cast<int>(result);
    }

    virtual ~cnnl_error() throw() {}

    virtual int get_error_number() const throw() { return error_number_; }
};

template <typename T>
::sycl::event copy(::sycl::queue &q, T *src, ::sycl::buffer<T, 1> &dst) {

    auto event = q.submit([&, src](::sycl::handler &cgh) {
        // Retrieve a  write accessor to a global buffer
        auto acc = dst.template get_access<::sycl::access::mode::write,
                impl::sycl::compat::target_device>(cgh);
        // Copy from the input pointer into the buffer associated with the
        // accessor
        cgh.copy(src, acc);
    });
    return event;
}

template <typename T>
::sycl::event copy(::sycl::queue &q, ::sycl::buffer<T, 1> &src, T *dst) {

    auto event = q.submit([&, dst](::sycl::handler &cgh) {
        // Retrieve a read accessor to a global buffer
        auto acc = src.template get_access<::sycl::access::mode::read,
                impl::sycl::compat::target_device>(cgh);
        // Copy from the buffer associated with the accessor into the output
        // pointer
        cgh.copy(acc, dst);
    });

    return event;
}

template <typename T>
::sycl::event copy(::sycl::queue &q, ::sycl::buffer<T, 1> &src,
        ::sycl::buffer<T, 1> &dst) {
    auto event = q.submit([&](::sycl::handler &cgh) {
        auto src_acc
                = src.template get_access<::sycl::access::mode::read_write>(
                        cgh);
        auto dst_acc
                = dst.template get_access<::sycl::access::mode::read_write>(
                        cgh);
        cgh.copy(src_acc, dst_acc);
    });
    return event;
}

static status_t cnnl_to_dnnl_status(cnnlStatus_t bang_status) {
    switch (bang_status) {
        case CNNL_STATUS_SUCCESS: return status::success;
        case CNNL_STATUS_BAD_PARAM: return status::invalid_arguments;
        case CNNL_STATUS_NOT_SUPPORTED: return status::unimplemented;
        default: return status::runtime_error;
    }
}

static status_t bang_to_dnnl_status(CNresult bang_result) {
    switch (bang_result) {
        case CN_SUCCESS: return status::success;
        default: return status::runtime_error;
    }
}

#define BANG_ERROR_LOCATION __FILE__ " : " STRINGIFY(__LINE__)

#define BANG_EXECUTE_FUNC(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != CN_SUCCESS) { \
            throw bang_error(std::string("At :") \
                            + std::string(BANG_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err); \
        } \
    }

#define CNNL_EXECUTE_FUNC(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != CNNL_STATUS_SUCCESS) { \
            throw cnnl_error(std::string("At :") \
                            + std::string(BANG_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err); \
        } \
    }

#define BANG_EXECUTE_FUNC_V(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != CN_SUCCESS) { \
            std::cout << bang_error(std::string("At :") \
                            + std::string(BANG_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err) \
                                 .what() \
                      << std::endl; \
        } \
    }

#define CNNL_EXECUTE_FUNC_V(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != CNNL_STATUS_SUCCESS) { \
            std::cout << cnnl_error(std::string("At :") \
                            + std::string(BANG_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err) \
                                 .what() \
                      << std::endl; \
        } \
    }

#define CNNL_CHECK_V(e) \
    { \
        auto status = (e); \
        if (status != CNNL_STATUS_SUCCESS) { \
            std::cout << cnnl_error(std::string("At :") \
                            + std::string(BANG_ERROR_LOCATION) \
                            + std::string(" : "), \
                    status) \
                                 .what() \
                      << std::endl; \
        } \
    }

#define BANG_EXECUTE_FUNC_S(name, ...) \
    [&]() { \
        auto err = name(__VA_ARGS__); \
        return bang_to_dnnl_status(err); \
    }()

#define CNNL_EXECUTE_FUNC_S(name, ...) \
    [&]() { \
        auto err = name(__VA_ARGS__); \
        if (err != CNNL_STATUS_SUCCESS) { return cnnl_to_dnnl_status(err); } \
        return status::success; \
    }()

static status_t create_and_set_tensor_descriptor(
        cnnlTensorDescriptor_t *tensor_desc, cnnlDataType_t data_type,
        int ndims, int *dims, int *strides) {

    CHECK(CNNL_EXECUTE_FUNC_S(cnnlCreateTensorDescriptor, tensor_desc));

    CHECK(CNNL_EXECUTE_FUNC_S(cnnlSetTensorDescriptorEx, *tensor_desc,
            cnnlTensorLayout_t::CNNL_LAYOUT_ARRAY, data_type, ndims, dims, strides));

    return status::success;
}

static status_t create_and_set_tensor_descriptor_ex(
        cnnlTensorDescriptor_t *tensor_desc, cnnlTensorLayout_t layout, cnnlDataType_t data_type,
        int ndims, int *dims) {

    CHECK(CNNL_EXECUTE_FUNC_S(cnnlCreateTensorDescriptor, tensor_desc));

    CHECK(CNNL_EXECUTE_FUNC_S(cnnlSetTensorDescriptor, *tensor_desc,
            layout, data_type, ndims, dims));

    return status::success;
}

static status_t create_and_set_conv_descriptor(
        cnnlConvolutionDescriptor_t *conv_desc, int ndims, int *padding,
        int *strides, int *dilation, int group_count,
        cnnlDataType_t data_type) {
    CHECK(CNNL_EXECUTE_FUNC_S(cnnlCreateConvolutionDescriptor, conv_desc));

    CHECK(CNNL_EXECUTE_FUNC_S(cnnlSetConvolutionDescriptor, *conv_desc,
            ndims, padding, strides, dilation, group_count, data_type));

    return status::success;
}

bool attr_post_ops_ok(const primitive_attr_t *attr);

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
