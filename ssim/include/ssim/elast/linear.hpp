#pragma once
#include "ssim/elast/basic_elast.hpp"
#include "ssim/elast/internal_physic.hpp"

namespace ssim::elast {
template <typename Scalar, typename Device, index_t Ndim>
class linear : public basic_elast_model<linear<Scalar, Device, Ndim>, Scalar, Device, Ndim> {
public:
  using base = basic_elast_model<linear<Scalar, Device, Ndim>, Scalar, Device, Ndim>;
  friend base;
  using scalar_type = typename base::scalar_type;
  using device_type = typename base::device_type;
  using def_grad_type = typename base::def_grad_type;
  using stress_type = typename base::stress_type;
  using hessian_type = typename base::hessian_type;
  using svd_matrix_type = typename base::svd_matrix_type;
  using svd_sigma_type = typename base::svd_sigma_type;
  static constexpr index_t ndim = base::ndim;

  linear(Scalar lambda, Scalar mu) : base(lambda, mu) {}
  SSIM_INTERNAL_ENABLE_ALL_CTOR(linear);

private:
  SSIM_PRIMFUNC scalar_type energy_impl(const def_grad_type& F,             //
                                        const svd_matrix_type& /* U */,     //
                                        const svd_sigma_type& /* sigma */,  //
                                        const svd_matrix_type& /* V */,
                                        Scalar lambda,
                                        Scalar mu) const noexcept {
    const Eigen::Matrix<Scalar, Ndim, Ndim> def_grad = mp::eigen_support::cmap(F);
    const auto approx_green_strain = internal::approx_green_strain(def_grad);
    const auto trace = approx_green_strain.trace();
    auto norm2_strain = approx_green_strain.squaredNorm();
    return mu * norm2_strain + 0.5 * lambda * trace * trace;
  }

  SSIM_PRIMFUNC scalar_type stress_impl(const stress_type& out,             //
                                        const def_grad_type& F,             //
                                        const svd_matrix_type& /* U */,     //
                                        const svd_sigma_type& /* sigma */,  //
                                        const svd_matrix_type& /* V */,
                                        Scalar lambda,
                                        Scalar mu) const noexcept {
    const Eigen::Matrix<Scalar, Ndim, Ndim> def_grad = mp::eigen_support::cmap(F);
    const auto green_strain = internal::approx_green_strain(def_grad);
    auto stress = mp::eigen_support::cmap(out);
    stress.noalias() = 2 * mu * green_strain +  //
                       lambda * green_strain.trace() * Eigen::Matrix<Scalar, Ndim, Ndim>::Identity();
    auto norm2_strain = green_strain.squaredNorm();
    const auto trace = green_strain.trace();
    return mu * norm2_strain + 0.5 * lambda * trace * trace;
  }

  SSIM_PRIMFUNC void hessian_impl(const hessian_type& out,            //
                                  const def_grad_type& /* F */,       //
                                  const svd_matrix_type& /* U */,     //
                                  const svd_sigma_type& /* sigma */,  //
                                  const svd_matrix_type& /* V */,
                                  Scalar lambda,
                                  Scalar mu) const noexcept {
    auto hessian = mp::eigen_support::cmap(out);
    hessian = mu * Eigen::Matrix<Scalar, Ndim * Ndim, Ndim * Ndim>::Identity();

    for (index_t i = 0; i < ndim; ++i) {
      for (index_t j = 0; j < ndim; ++j) {
        hessian(i * Ndim + j, j * Ndim + i) += mu;
        hessian(i * Ndim + i, j * Ndim + j) += lambda;
      }
    }
  }
};
}  // namespace ssim::elast
