#pragma once
#include <mathprim/core/view.hpp>

#include "ssim/defines.hpp"
namespace ssim::elast {

/**
 * @brief Convert young's modulus (E) and poisson's ratio (nu) to Lamé parameters(second)
 */
template <typename Scalar>
Scalar compute_mu(Scalar E, Scalar nu) {
  return E / (2 * (1 + nu));
}
/**
 * @brief Convert young's modulus (E) and poisson's ratio (nu) to Lamé parameters(first)
 */
template <typename Scalar>
Scalar compute_lambda(Scalar E, Scalar nu) {
  return E * nu / ((1 + nu) * (1 - 2 * nu));
}

/// @brief Deformation gradient
template <typename Scalar, typename Device, index_t Ndim>
using def_grad_t = mp::contiguous_view<Scalar, mp::shape_t<Ndim, Ndim>, Device>;

/// @brief Matrix of SVD
template <typename Scalar, typename Device, index_t Ndim>
using svd_matrix_t = mp::contiguous_view<Scalar, mp::shape_t<Ndim, Ndim>, Device>;

/// @brief Singular values of SVD
template <typename Scalar, typename Device, index_t Ndim>
using svd_sigma_t = mp::contiguous_view<const Scalar, mp::shape_t<Ndim>, Device>;

/// @brief Stress tensor
template <typename Scalar, typename Device, index_t Ndim>
using stress_t = mp::contiguous_view<Scalar, mp::shape_t<Ndim, Ndim>, Device>;

/// @brief Hessian tensor
template <typename Scalar, typename Device, index_t Ndim>
using hessian_t = mp::contiguous_view<Scalar, mp::shape_t<Ndim * Ndim, Ndim * Ndim>, Device>;

template <typename Derived, typename Scalar, typename Device, index_t Ndim>
class basic_elast_model {
public:
  using scalar_type = Scalar;
  using device_type = Device;
  using def_grad_type = def_grad_t<const Scalar, Device, Ndim>;
  using stress_type = stress_t<Scalar, Device, Ndim>;
  using hessian_type = hessian_t<Scalar, Device, Ndim>;
  using svd_matrix_type = svd_matrix_t<const Scalar, Device, Ndim>;
  using svd_sigma_type = svd_sigma_t<const Scalar, Device, Ndim>;
  static constexpr index_t ndim = Ndim;

  basic_elast_model(Scalar lambda, Scalar mu) : lambda_(lambda), mu_(mu) {}
  basic_elast_model(const basic_elast_model&) = default;
  basic_elast_model(basic_elast_model&&) = default;
  basic_elast_model& operator=(const basic_elast_model&) = default;
  basic_elast_model& operator=(basic_elast_model&&) = default;

  SSIM_PRIMFUNC scalar_type energy(const def_grad_type& F,       //
                                   const svd_matrix_type& U,     //
                                   const svd_sigma_type& sigma,  //
                                   const svd_matrix_type& V) const noexcept {
    return static_cast<const Derived*>(this)->energy_impl(F, U, sigma, V, lambda_, mu_);
  }

  SSIM_PRIMFUNC scalar_type energy(const def_grad_type& F,       //
                                   const svd_matrix_type& U,     //
                                   const svd_sigma_type& sigma,  //
                                   const svd_matrix_type& V,     //
                                   Scalar lambda,                //
                                   Scalar mu) const noexcept {
    return static_cast<const Derived*>(this)->energy_impl(F, U, sigma, V, lambda, mu);
  }

  SSIM_PRIMFUNC scalar_type stress(const stress_type& out,       //
                                   const def_grad_type& F,       //
                                   const svd_matrix_type& U,     //
                                   const svd_sigma_type& sigma,  //
                                   const svd_matrix_type& V) const noexcept {
    return static_cast<const Derived*>(this)->stress_impl(out, F, U, sigma, V, lambda_, mu_);
  }

  SSIM_PRIMFUNC scalar_type stress(const stress_type& out,       //
                                   const def_grad_type& F,       //
                                   const svd_matrix_type& U,     //
                                   const svd_sigma_type& sigma,  //
                                   const svd_matrix_type& V, Scalar lambda, Scalar mu) const noexcept {
    return static_cast<const Derived*>(this)->stress_impl(out, F, U, sigma, V, lambda, mu);
  }

  SSIM_PRIMFUNC void hessian(const hessian_type& out,      //
                             const def_grad_type& F,       //
                             const svd_matrix_type& U,     //
                             const svd_sigma_type& sigma,  //
                             const svd_matrix_type& V) const noexcept {
    static_cast<const Derived*>(this)->hessian_impl(out, F, U, sigma, V, lambda_, mu_);
  }

  SSIM_PRIMFUNC void hessian(const hessian_type& out,      //
                             const def_grad_type& F,       //
                             const svd_matrix_type& U,     //
                             const svd_sigma_type& sigma,  //
                             const svd_matrix_type& V,     //
                             Scalar lambda,                //
                             Scalar mu) const noexcept {
    static_cast<const Derived*>(this)->hessian_impl(out, F, U, sigma, V, lambda, mu);
  }

  Scalar lambda_;
  Scalar mu_;
};

template <typename ElastImpl>
class energy_operator {
  using scalar_type = typename ElastImpl::scalar_type;
  using device_type = typename ElastImpl::device_type;
  using def_grad_type = typename ElastImpl::def_grad_type;
  using stress_type = typename ElastImpl::stress_type;
  using hessian_type = typename ElastImpl::hessian_type;
  using svd_matrix_type = typename ElastImpl::svd_matrix_type;
  using svd_sigma_type = typename ElastImpl::svd_sigma_type;
  static constexpr index_t ndim = ElastImpl::ndim;

  SSIM_PRIMFUNC explicit energy_operator(ElastImpl model) : model_(model) {}
  SSIM_INTERNAL_ALL_ENABLE(energy_operator);

  SSIM_PRIMFUNC scalar_type operator()(const def_grad_type& F,       //
                                       const svd_matrix_type& U,     //
                                       const svd_sigma_type& sigma,  //
                                       const svd_matrix_type& V) const noexcept {
    return model_.energy(F, U, sigma, V);
  }

  SSIM_PRIMFUNC scalar_type operator()(const def_grad_type& F) const noexcept {  //
    return model_.energy(F, {}, {}, {});
  }

  ElastImpl model_;
};

template <typename ElastImpl>
class stress_operator {
  using scalar_type = typename ElastImpl::scalar_type;
  using device_type = typename ElastImpl::device_type;
  using def_grad_type = typename ElastImpl::def_grad_type;
  using stress_type = typename ElastImpl::stress_type;
  using hessian_type = typename ElastImpl::hessian_type;
  using svd_matrix_type = typename ElastImpl::svd_matrix_type;
  using svd_sigma_type = typename ElastImpl::svd_sigma_type;
  static constexpr index_t ndim = ElastImpl::ndim;

  SSIM_PRIMFUNC explicit stress_operator(ElastImpl model) : model_(model) {}
  SSIM_INTERNAL_ALL_ENABLE(stress_operator);

  SSIM_PRIMFUNC scalar_type operator()(const stress_type& out,       //
                                       const def_grad_type& F,       //
                                       const svd_matrix_type& U,     //
                                       const svd_sigma_type& sigma,  //
                                       const svd_matrix_type& V) const noexcept {
    return model_.stress(out, F, U, sigma, V);
  }

  SSIM_PRIMFUNC scalar_type operator()(const stress_type& out,  //
                                       const def_grad_type& F) const noexcept {
    return model_.stress(out, F, {}, {}, {});
  }

  ElastImpl model_;
};

template <typename ElastImpl>
class hessian_operator {
  using scalar_type = typename ElastImpl::scalar_type;
  using device_type = typename ElastImpl::device_type;
  using def_grad_type = typename ElastImpl::def_grad_type;
  using stress_type = typename ElastImpl::stress_type;
  using hessian_type = typename ElastImpl::hessian_type;
  using svd_matrix_type = typename ElastImpl::svd_matrix_type;
  using svd_sigma_type = typename ElastImpl::svd_sigma_type;
  static constexpr index_t ndim = ElastImpl::ndim;

  SSIM_PRIMFUNC explicit hessian_operator(ElastImpl model) : model_(model) {}
  SSIM_INTERNAL_ALL_ENABLE(hessian_operator);

  SSIM_PRIMFUNC scalar_type operator()(const hessian_type& out,      //
                                       const def_grad_type& F,       //
                                       const svd_matrix_type& U,     //
                                       const svd_sigma_type& sigma,  //
                                       const svd_matrix_type& V) const noexcept {
    return model_.hessian(out, F, U, sigma, V);
  }

  SSIM_PRIMFUNC scalar_type operator()(const hessian_type& out,  //
                                       const def_grad_type& F) const noexcept {
    return model_.hessian(out, F, {}, {}, {});
  }

  ElastImpl model_;
};

}  // namespace ssim::elast
