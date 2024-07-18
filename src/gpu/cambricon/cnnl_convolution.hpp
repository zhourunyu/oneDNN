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

#ifndef GPU_CAMBRICON_CNNL_CONVOLUTION_HPP
#define GPU_CAMBRICON_CNNL_CONVOLUTION_HPP

#include "cnnl.h"

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc.hpp"
#include "gpu/cambricon/cnnl_convolution_impl.hpp"
#include "gpu/cambricon/cnnl_convolution_pd.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/cambricon/sycl_bang_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct cnnl_convolution_fwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public cnnl_convolution_fwd_pd_t {
        using cnnl_convolution_fwd_pd_t::cnnl_convolution_fwd_pd_t;
        pd_t(const pd_t &other)
            : cnnl_convolution_fwd_pd_t(other)
            , impl_(other.impl_)
            , dst_md_temp_(other.dst_md_temp_) {}

        DECLARE_COMMON_PD_T("bang:cnnl:any", cnnl_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;

            using sm_t = primitive_attr_t::skip_mask_t;
            const auto attr_skip_mask = sm_t::scales_runtime | sm_t::post_ops;
            auto *sycl_engine
                    = utils::downcast<impl::sycl::sycl_engine_base_t *>(engine);

            bool ok = utils::one_of(desc()->prop_kind,
                    prop_kind::forward_training, prop_kind::forward_inference);
            ok = ok && attr()->has_default_values(attr_skip_mask);
            ok = ok && attr_post_ops_ok(attr());
            ok = ok
                    && (utils::everyone_is(f32, src_md_.data_type,
                                weights_md_.data_type, dst_md_.data_type)
                            || utils::everyone_is(f16, src_md_.data_type,
                                    weights_md_.data_type, dst_md_.data_type)
                            || utils::everyone_is(bf16, src_md_.data_type,
                                    weights_md_.data_type, dst_md_.data_type)
                            || (utils::everyone_is(s8, src_md_.data_type,
                                        weights_md_.data_type)
                                    && utils::one_of(
                                            dst_md_.data_type, f32, s8)))
                    && IMPLICATION(
                            utils::one_of(data_type::bf16, src_md_.data_type,
                                    weights_md_.data_type, dst_md_.data_type),
                            has_bf16_support(sycl_engine->device()));

            ok = ok && this->set_default_formats();
            ok = ok
                    && IMPLICATION(
                            desc()->alg_kind == dnnl_convolution_winograd,
                            ndims() < 5 && src_md_.data_type != s8);
            ok = ok
                    && IMPLICATION(!attr()->scales_.has_default_values(),
                            utils::one_of(src_md_.data_type, s8)
                                    && attr_scales_ok());
            ok = ok
                    && IMPLICATION(
                            src_md_.data_type == s8, check_s8_configuration())
                    // cnnlAssignAdd used for bias add requires both tensors
                    // to have the same data type.
                    && IMPLICATION(with_bias(),
                            dst_md_.data_type == bias_md_.data_type);
            ok = ok && memory_format_ok(&src_md_);
            ok = ok && memory_format_ok(&weights_md_);
            ok = ok && memory_format_ok(&dst_md_);
            if (with_bias()) ok = ok && memory_format_ok(&bias_md_);
            if (!ok) return status::unimplemented;

            if (check_for_zero_dims()) return status::success;

            const bool use_temp_dst = attr()->post_ops_.len() > 0;
            if (use_temp_dst) {
                dst_md_temp_ = dst_md_;
                if (dst_md_.data_type == s8) { dst_md_temp_.data_type = f32; }
            }

            impl_.reset(new cnnl_convolution_impl_fwd_t());
            return impl_->init(engine, this, use_temp_dst);
        }
        bool with_scratchpad() const { return impl_->with_scratchpad(); }
        std::shared_ptr<cnnl_convolution_impl_base_t> impl_;
        memory_desc_t dst_md_temp_;

        bool use_temp_dst() const {
            if (impl_.get()) return impl_->use_temp_dst();
            return false;
        }

    private:
        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, gowi, gohwi, godhwi)
                    : utils::pick(ndims() - 3, owi, ohwi, odhwi);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }

        bool check_s8_configuration() const {
            const auto check_nhwc
                    = [](const memory_desc_t &md, bool is_weights = false) {
                          cnnlTensorLayout_t fmt;
                          get_format(&md, fmt, is_weights);
                          return fmt == CNNL_LAYOUT_NHWC;
                      };

            return check_nhwc(src_md_) && check_nhwc(dst_md_)
                    && check_nhwc(weights_md_, true)
                    && (src_md_.dims[1] % 4) == 0 && (dst_md_.dims[1] % 4) == 0
                    && ndims() < 5;
        }

        bool attr_scales_ok() const {
            const auto &scales = attr()->scales_;
            const auto &supported_args
                    = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};
            if (!scales.has_default_values(supported_args)) return false;
            // cnnl does not support scaling per dimension.
            for (auto arg : supported_args)
                if (scales.get(arg).mask_ != 0) return false;
            return true;
        }
    };

    status_t init_temp_dst(engine_t *engine) {
        auto sycl_engine = utils::downcast<sycl_bang_engine_t *>(engine);
        memory_storage_t *scratch_ptr = nullptr;
        auto wrap = memory_desc_wrapper(pd()->dst_md_temp_);
        CHECK(sycl_engine->create_memory_storage(
                &scratch_ptr, memory_flags_t::alloc, wrap.size(), nullptr));
        scratch_storage.reset(scratch_ptr);

        return status::success;
    }

    virtual status_t init(engine_t *engine) override {
        const auto impl = pd()->impl_.get();
        if (impl && impl->use_temp_dst()) { init_temp_dst(engine); }
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->check_for_zero_dims()) { return status::success; }

        execute_convolution(ctx, pd()->with_bias(), pd()->with_scratchpad());

        return status::success;
    }
    status_t execute_convolution(
            const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const;

private:
    ::sycl::buffer<uint8_t, 1> &buffer(memory_storage_t *mem_storage) const {
        return utils::downcast<impl::sycl::sycl_buffer_memory_storage_t *>(
                mem_storage)
                ->buffer();
    }
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<memory_storage_t> scratch_storage;
};

struct cnnl_convolution_bwd_data_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public cnnl_convolution_bwd_data_pd_t {
        using cnnl_convolution_bwd_data_pd_t::cnnl_convolution_bwd_data_pd_t;

        DECLARE_COMMON_PD_T("bang:cnnl:any", cnnl_convolution_bwd_data_t);

        status_t init(engine_t *engine) {
            using namespace data_type;

            bool ok = desc()->prop_kind == prop_kind::backward_data;
            auto *sycl_engine
                    = utils::downcast<impl::sycl::sycl_engine_base_t *>(engine);
            ok = ok && this->set_default_formats();
            ok = ok
                    && (utils::everyone_is(f32, diff_src_md_.data_type,
                                weights_md_.data_type, diff_dst_md_.data_type)
                            || utils::everyone_is(f16, diff_src_md_.data_type,
                                    weights_md_.data_type,
                                    diff_dst_md_.data_type)
                            || utils::everyone_is(bf16, diff_src_md_.data_type,
                                    weights_md_.data_type,
                                    diff_dst_md_.data_type))
                    && IMPLICATION(utils::one_of(data_type::bf16,
                                           diff_src_md_.data_type,
                                           weights_md_.data_type,
                                           diff_dst_md_.data_type),
                            has_bf16_support(sycl_engine->device()));

            ok = ok
                    && IMPLICATION(
                            desc()->alg_kind == dnnl_convolution_winograd,
                            ndims() < 5);
            ok = ok && memory_format_ok(&diff_src_md_);
            ok = ok && memory_format_ok(&weights_md_);
            ok = ok && memory_format_ok(&diff_dst_md_);
            if (with_bias()) {
                ok = ok && memory_format_ok(&bias_md_);
                ok = ok && bias_md_.data_type == diff_dst_md_.data_type;
            }
            if (!ok) return status::unimplemented;

            if (check_for_zero_dims()) return status::success;

            impl_.reset(new cnnl_convolution_impl_bwd_data_t());
            return impl_->init(engine, this);
        }

        std::shared_ptr<cnnl_convolution_impl_base_t> impl_;

        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, gowi, gohwi, godhwi)
                    : utils::pick(ndims() - 3, owi, ohwi, odhwi);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
        bool with_scratchpad() const { return impl_->with_scratchpad(); }
        bool support_bias() const override { return true; }
    };

    ~cnnl_convolution_bwd_data_t() {}
    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->check_for_zero_dims()) { return status::success; }
        return execute_convolution(
                ctx, pd()->with_bias(), pd()->with_scratchpad());
    }
    status_t execute_convolution(
            const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct cnnl_convolution_bwd_weights_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public cnnl_convolution_bwd_weights_pd_t {
        using cnnl_convolution_bwd_weights_pd_t::
                cnnl_convolution_bwd_weights_pd_t;

        DECLARE_COMMON_PD_T("bang:cnnl:any", cnnl_convolution_bwd_weights_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            bool ok = desc()->prop_kind == prop_kind::backward_weights;
            auto *sycl_engine
                    = utils::downcast<impl::sycl::sycl_engine_base_t *>(engine);
            ok = ok && this->set_default_formats();
            ok = ok
                    && (utils::everyone_is(f32, src_md_.data_type,
                                diff_weights_md_.data_type,
                                diff_dst_md_.data_type)
                            || utils::everyone_is(f16, src_md_.data_type,
                                    diff_weights_md_.data_type,
                                    diff_dst_md_.data_type)
                            || utils::everyone_is(bf16, src_md_.data_type,
                                    diff_weights_md_.data_type,
                                    diff_dst_md_.data_type))

                    && IMPLICATION(
                            utils::one_of(data_type::bf16, src_md_.data_type,
                                    diff_weights_md_.data_type,
                                    diff_dst_md_.data_type),
                            has_bf16_support(sycl_engine->device())
                                    && !with_bias());

            ok = ok
                    && IMPLICATION(
                            desc()->alg_kind == dnnl_convolution_winograd,
                            ndims() < 5);
            ok = ok && memory_format_ok(&src_md_);
            ok = ok && memory_format_ok(&diff_weights_md_);
            ok = ok && memory_format_ok(&diff_dst_md_);
            if (with_bias()) {
                ok = ok && memory_format_ok(&diff_bias_md_);
                ok = ok && diff_bias_md_.data_type == diff_dst_md_.data_type;
            }
            if (!ok) return status::unimplemented;

            impl_.reset(new cnnl_convolution_impl_bwd_weights_t());
            if (check_for_zero_dims()) { return impl_->init_zero_dims(this); };

            return impl_->init(engine, this);
        }

        std::shared_ptr<cnnl_convolution_impl_base_t> impl_;

        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, gowi, gohwi, godhwi)
                    : utils::pick(ndims() - 3, owi, ohwi, odhwi);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
        bool with_scratchpad() const { return impl_->with_scratchpad(); }
    };

    ~cnnl_convolution_bwd_weights_t() {}
    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->check_for_zero_dims()) { return execute_zero_dims(ctx); }
        return execute_convolution(
                ctx, pd()->with_bias(), pd()->with_scratchpad());
    }
    status_t execute_convolution(
            const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const;
    status_t execute_zero_dims(const exec_ctx_t &ctx) const;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
