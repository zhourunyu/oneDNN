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
#include "common/engine.hpp"
#include "common/impl_list_item.hpp"
#include "gpu/cambricon/cnnl_reorder.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/ocl/cross_engine_reorder.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

namespace {

#define REORDER_INSTANCE(...) \
    impl_list_item_t( \
            impl_list_item_t::reorder_type_deduction_helper_t<__VA_ARGS__>()),

// clang-format off
constexpr impl_list_item_t bang_reorder_impl_list[] = {
        REORDER_INSTANCE(gpu::ocl::cross_engine_reorder_t::pd_t)
        REORDER_INSTANCE(cnnl_reorder_t::pd_t)
        nullptr,
};
// clang-format on

} // namespace

const impl_list_item_t *
bang_gpu_engine_impl_list_t::get_reorder_implementation_list(
        const memory_desc_t *, const memory_desc_t *) {
    return bang_reorder_impl_list;
}

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl
