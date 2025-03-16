#pragma once
#include "mathprim/core/defines.hpp"

#define SSIM_FORCE_INLINE MATHPRIM_FORCE_INLINE
#define SSIM_NOINLINE MATHPRIM_NOINLINE

#define SSIM_HOST MATHPRIM_HOST
#define SSIM_DEVICE MATHPRIM_DEVICE
#define SSIM_GENERAL MATHPRIM_GENERAL
#define SSIM_PRIMFUNC MATHPRIM_PRIMFUNC
#define SSIM_ASSERT(cond) MATHPRIM_ASSERT(cond)
#define SSIM_CONSTEXPR MATHPRIM_CONSTEXPR
#define SSIM_UNUSED(x) MATHPRIM_UNUSED(x)

#define SSIM_INTERNAL_COPY(t, enable) MATHPRIM_INTERNAL_COPY(t, enable)
#define SSIM_INTERNAL_MOVE(t, enable) MATHPRIM_INTERNAL_MOVE(t, enable)
#define SSIM_INTERNAL_ENABLE_ALL_CTOR(t) \
  SSIM_INTERNAL_COPY(t, default);      \
  SSIM_INTERNAL_MOVE(t, default)

namespace ssim {

namespace mp = ::mathprim;

using size_t = mp::size_t;
using index_t = mp::index_t;
using int8_t = std::int8_t;

} // namespace ssim
