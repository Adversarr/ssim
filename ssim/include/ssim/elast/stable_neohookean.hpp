#pragma once
#include "ssim/elast/basic_elast.hpp"
#include "ssim/elast/internal_physic.hpp"

namespace ssim::elast {
template <typename Scalar, typename Device, index_t Ndim>
class stable_neohookean : public basic_elast_model<stable_neohookean<Scalar, Device, Ndim>, Scalar, Device, Ndim> {
public:
  using base = basic_elast_model<stable_neohookean<Scalar, Device, Ndim>, Scalar, Device, Ndim>;
  friend base;
  using scalar_type = typename base::scalar_type;
  using device_type = typename base::device_type;
  using def_grad_type = typename base::def_grad_type;
  using stress_type = typename base::stress_type;
  using hessian_type = typename base::hessian_type;
  using svd_matrix_type = typename base::svd_matrix_type;
  using svd_sigma_type = typename base::svd_sigma_type;
  static constexpr index_t ndim = base::ndim;
  stable_neohookean(Scalar lambda, Scalar mu) : base(lambda, mu) {}
  SSIM_INTERNAL_ENABLE_ALL_CTOR(stable_neohookean);

private:
  SSIM_PRIMFUNC scalar_type energy_impl(const def_grad_type& F,             //
                                        const svd_matrix_type& /* U */,     //
                                        const svd_sigma_type& /* sigma */,  //
                                        const svd_matrix_type& /* V */,     //
                                        Scalar lambda,                      //
                                        Scalar mu) const noexcept {
    auto def_grad = mp::eigen_support::cmap(F);
    const Scalar Ic = def_grad.squaredNorm();
    const Scalar del = delta(lambda, mu);
    const Scalar alpha = (1 - 1 / (Ndim + del)) * mu / lambda + 1;
    const Scalar Jminus1 = def_grad.determinant() - alpha;
    const Scalar zero_energy = 0.5 * (lambda * (1 - alpha) * (1 - alpha) - mu * log(Ndim + del));
    return 0.5 * (mu * (Ic - Ndim) + lambda * Jminus1 * Jminus1 - mu * log(Ic + del)) - zero_energy;
  }

  SSIM_PRIMFUNC scalar_type stress_impl(const stress_type& out,             //
                                        const def_grad_type& F,             //
                                        const svd_matrix_type& /* U */,     //
                                        const svd_sigma_type& /* sigma */,  //
                                        const svd_matrix_type& /* V */,     //
                                        Scalar lambda,                      //
                                        Scalar mu) const noexcept {
    // Real mu = this->mu_, lambda = this->lambda_;
    // Real const Ic = math::norm2(F);
    // Real const del= delta();
    // Real alpha = (1 - 1 / (dim + del)) * mu / lambda + 1;
    // Real const Jminus1 = math::det(F) - alpha;
    // math::RealMatrix<dim, dim> dJdF = details::partial_determinant(F);
    // return lambda * Jminus1 * dJdF  + mu * (1.0 - 1.0 / (Ic + del)) * F;
    const Eigen::Matrix<Scalar, Ndim, Ndim> def_grad = mp::eigen_support::cmap(F);
    auto stress = mp::eigen_support::cmap(out);
    const Scalar Ic = def_grad.squaredNorm();
    const Scalar del = delta(lambda, mu);
    const Scalar alpha = (1 - 1 / (Ndim + del)) * mu / lambda + 1;
    const Scalar Jminus1 = def_grad.determinant() - alpha;
    auto dJdF = internal::partial_determinant(def_grad);
    stress.noalias() = lambda * Jminus1 * dJdF + mu * (1.0 - 1.0 / (Ic + del)) * def_grad;
    const Scalar zero_energy = 0.5 * (lambda * (1 - alpha) * (1 - alpha) - mu * log(Ndim + del));
    return 0.5 * (mu * (Ic - Ndim) + lambda * Jminus1 * Jminus1 - mu * log(Ic + del)) - zero_energy;
  }

  SSIM_PRIMFUNC void hessian_impl(const hessian_type& out,            //
                                  const def_grad_type& F,             //
                                  const svd_matrix_type& /* U */,     //
                                  const svd_sigma_type& /* sigma */,  //
                                  const svd_matrix_type& /* V */,     //
                                  Scalar lambda,                      //
                                  Scalar mu) const noexcept {
    const Eigen::Matrix<Scalar, Ndim, Ndim> def_grad = mp::eigen_support::cmap(F);
    const Scalar I2 = def_grad.squaredNorm();
    const Scalar del = delta(lambda, mu);
    const Scalar I3 = def_grad.determinant();
    const Scalar alpha = (1 - 1 / (Ndim + del)) * mu / lambda + 1;
    const Scalar scale = lambda * (I3 - alpha);
    auto dJdF_matrix = internal::partial_determinant(def_grad);
    
    Eigen::Vector<Scalar, Ndim * Ndim> dJdF, F_flat;
    for (index_t i = 0; i < Ndim * Ndim; ++i) {
      dJdF(i) = dJdF_matrix(i % Ndim, i / Ndim);
      F_flat(i) = def_grad(i % Ndim, i / Ndim);
    }
    Eigen::Matrix<Scalar, Ndim*Ndim, Ndim*Ndim> hess;
    hess.setIdentity(); hess *= mu * (1 - 1 / (I2 + del));
    // hess = mu * (1 - 1 / (I2 + del)) * Eigen::Matrix<Scalar, Ndim * Ndim, Ndim * Ndim>::Identity();
    hess.noalias() += mu * (2.0 / (I2 + del) / (I2 + del)) * (F_flat * F_flat.transpose());
    hess.noalias() += lambda * dJdF * dJdF.transpose();
    internal::add_HJ(hess, def_grad, scale);
    mp::eigen_support::cmap(out) = hess;
  }

  SSIM_PRIMFUNC Scalar delta(Scalar lambda, Scalar mu) const noexcept {
    if constexpr (Ndim == 2) {
      const Scalar t = mu / lambda;
      const Scalar a = 2 * t + 1;                  // > 0, 2t + 1.
      const Scalar b = 2 * t * (Ndim - 1) + Ndim;  // > 0, 2t + 2.
      const Scalar c = -t * Ndim;                  // < 0, -2t.
      const Scalar delta = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
      return delta;
    } else /* Ndim == 3 */ {
      return 1;
    }
  }

  Scalar approximated_diffusivity_impl() const noexcept { 
    Scalar lambda = this->lambda_, mu = this->mu_;
    Scalar true_lambda = lambda + static_cast<Scalar>(5. / 6.) * mu;
    Scalar true_mu = static_cast<Scalar>(4. / 3.) * mu;
    return true_lambda + 2 * true_mu;
  }
};
}  // namespace ssim::elast
