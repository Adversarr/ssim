#pragma once
#include <mathprim/sparse/basic_sparse.hpp>

#include "ssim/defines.hpp"  // IWYU pragma: export
namespace ssim::fem {

/// currently, we only support CSR format
template <typename Scalar, typename Device>
using sparse_matrix = mp::sparse::basic_sparse_matrix<Scalar, Device, mp::sparse::sparse_format::csr>;
template <typename Scalar, typename Device>
using sparse_view = mp::sparse::basic_sparse_view<Scalar, Device, mp::sparse::sparse_format::csr>;

}  // namespace ssim::fem