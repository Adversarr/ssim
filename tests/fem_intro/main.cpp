#include <gtest/gtest.h>

#include "ssim/elast/linear.hpp"
#include "ssim/elast/stable_neohookean.hpp"
#include "ssim/mesh/basic_mesh.hpp"

using namespace ssim;

GTEST_TEST(elast, linear3) {
  using S = double;
  static constexpr index_t ndim = 3;
  using D = mp::device::cpu;
  const S youngs = 1e6, poisson = 0.4;
  const S lambda = elast::compute_lambda(youngs, poisson);
  const S mu = elast::compute_mu(youngs, poisson);
  elast::linear<S, D, ndim> elast(lambda, mu);

  auto F = Eigen::Matrix<S, ndim, ndim>::Identity().eval();
  auto F_view = mp::eigen_support::view(F);
  auto dE_dF = Eigen::Matrix<S, ndim, ndim>::Zero().eval();
  auto dE_dF_view = mp::eigen_support::view(dE_dF);

  auto energy = elast.stress(dE_dF_view, F_view, {}, {}, {});
  EXPECT_NEAR(energy, 0.0, 1e-7);
  for (index_t i = 0; i < ndim; ++i) {
    for (index_t j = 0; j < ndim; ++j) {
      EXPECT_NEAR(dE_dF(i, j), 0.0, 1e-7);
    }
  }

  // We test the relationship between energy and stress by finite difference.
  F = Eigen::Matrix<S, ndim, ndim>{{1, 0.3, 0.2}, {0.2, 2, 0.1}, {0.1, 0.3, 1}};
  auto F_backup = F.eval();
  elast.stress(dE_dF_view, F_view, {}, {}, {});
  auto dE_dF_center = dE_dF.eval();

  S del = 1e-3;
  for (int i = 0; i < ndim; ++i) {
    for (int j = 0; j < ndim; ++j) {
      F = F_backup;
      F_view(i, j) += del;
      auto energy1 = elast.energy(F_view, {}, {}, {});
      F = F_backup;
      F_view(i, j) -= del;
      auto energy2 = elast.energy(F_view, {}, {}, {});
      EXPECT_NEAR(dE_dF_view(i, j), (energy1 - energy2) / (2 * del), 1e-3);
    }
  }

  // Test the relationship between Hessian and by finite difference.
  Eigen::Matrix<S, ndim * ndim, ndim * ndim> H;
  auto H_view = mp::eigen_support::view(H);
  elast.hessian(H_view, F_view, {}, {}, {});

  auto dE_dF_backup = dE_dF.eval();
  // std::cout << H << std::endl;
  // std::cout << (H - H.transpose()).squaredNorm() << std::endl;

  for (int i = 0; i < ndim; ++i) {
    for (int j = 0; j < ndim; ++j) {
      F = F_backup;
      F_view(i, j) += del;
      elast.stress(dE_dF_view, F_view, {}, {}, {});
      auto dE_dF_forward = dE_dF.eval();
      F = F_backup;
      F_view(i, j) -= del;
      elast.stress(dE_dF_view, F_view, {}, {}, {});
      auto dE_dF_backward = dE_dF.eval();

      auto fwd_view = mp::eigen_support::view(dE_dF_forward);
      auto bwd_view = mp::eigen_support::view(dE_dF_backward);

      for (auto [k, l] : mp::make_shape(ndim, ndim)) {
        S hes_elem = H(i * ndim + j, k * ndim + l);
        S finite_diff = (fwd_view(l, k) - bwd_view(l, k)) / (2 * del);
        EXPECT_NEAR(hes_elem, finite_diff, 1e-3);
      }
    }
  }
}

GTEST_TEST(elast, snh3) {
  using S = double;
  static constexpr index_t ndim = 3;
  using D = mp::device::cpu;
  const S youngs = 1e6, poisson = 0.3;
  const S lambda = elast::compute_lambda(youngs, poisson);
  const S mu = elast::compute_mu(youngs, poisson);
  elast::stable_neohookean<S, D, ndim> elast(lambda, mu);

  auto F = Eigen::Matrix<S, ndim, ndim>::Identity().eval();
  auto F_view = mp::eigen_support::view(F);
  auto dE_dF = Eigen::Matrix<S, ndim, ndim>::Zero().eval();
  auto dE_dF_view = mp::eigen_support::view(dE_dF);

  auto energy = elast.stress(dE_dF_view, F_view, {}, {}, {});
  EXPECT_NEAR(energy, 0.0, 1e-7);
  for (index_t i = 0; i < ndim; ++i) {
    for (index_t j = 0; j < ndim; ++j) {
      EXPECT_NEAR(dE_dF(i, j), 0.0, 1e-7);
    }
  }

  // We test the relationship between energy and stress by finite difference.
  F = Eigen::Matrix<S, ndim, ndim>{{1, 0.3, 0.2}, {0.2, 2, 0.1}, {0.1, 0.3, 1}};
  auto F_backup = F.eval();
  auto energy_center = elast.stress(dE_dF_view, F_view, {}, {}, {});
  auto dE_dF_center = dE_dF.eval();
  std::cout << dE_dF_center << std::endl;

  S del = 1e-6;
  for (int i = 0; i < ndim; ++i) {
    for (int j = 0; j < ndim; ++j) {
      F = F_backup;
      F_view(i, j) += del;
      auto energy1 = elast.energy(F_view, {}, {}, {});
      F = F_backup;
      F_view(i, j) -= del;
      auto energy2 = elast.energy(F_view, {}, {}, {});
      EXPECT_NEAR(dE_dF_view(i, j), (energy1 - energy2) / (2 * del), 1e-3);
    }
  }

  // Test the relationship between Hessian and by finite difference.
  Eigen::Matrix<S, ndim * ndim, ndim * ndim> H;
  auto H_view = mp::eigen_support::view(H);
  elast.hessian(H_view, F_view, {}, {}, {});

  auto dE_dF_backup = dE_dF.eval();

  for (int i = 0; i < ndim; ++i) {
    for (int j = 0; j < ndim; ++j) {
      F = F_backup;
      F_view(i, j) += del;
      elast.stress(dE_dF_view, F_view, {}, {}, {});
      auto dE_dF_forward = dE_dF.eval();
      F = F_backup;
      F_view(i, j) -= del;
      elast.stress(dE_dF_view, F_view, {}, {}, {});
      auto dE_dF_backward = dE_dF.eval();

      for (auto [k, l] : mp::make_shape(ndim, ndim)) {
        S hes_elem = H(i * ndim + j, k * ndim + l);
        S finite_diff = (dE_dF_forward(l, k) - dE_dF_backward(l, k)) / (2 * del);
        S err = std::abs(hes_elem - finite_diff);
        EXPECT_NEAR(err / std::max(std::abs(hes_elem), std::abs(finite_diff)), 0.0, 1e-3);
      }
    }
  }
}


GTEST_TEST(elast, linear2) {
  using S = double;
  static constexpr index_t ndim = 2;
  using D = mp::device::cpu;
  const S youngs = 1e6, poisson = 0.4;
  const S lambda = elast::compute_lambda(youngs, poisson);
  const S mu = elast::compute_mu(youngs, poisson);
  elast::linear<S, D, ndim> elast(lambda, mu);

  auto F = Eigen::Matrix<S, ndim, ndim>::Identity().eval();
  auto F_view = mp::eigen_support::view(F);
  auto dE_dF = Eigen::Matrix<S, ndim, ndim>::Zero().eval();
  auto dE_dF_view = mp::eigen_support::view(dE_dF);

  auto energy = elast.stress(dE_dF_view, F_view, {}, {}, {});
  EXPECT_NEAR(energy, 0.0, 1e-7);
  for (index_t i = 0; i < ndim; ++i) {
    for (index_t j = 0; j < ndim; ++j) {
      EXPECT_NEAR(dE_dF(i, j), 0.0, 1e-7);
    }
  }

  // We test the relationship between energy and stress by finite difference.
  F = Eigen::Matrix<S, ndim, ndim>{{1, 0.3}, {0.2, 3}};
  auto F_backup = F.eval();
  elast.stress(dE_dF_view, F_view, {}, {}, {});
  auto dE_dF_center = dE_dF.eval();

  S del = 1e-3;
  for (int i = 0; i < ndim; ++i) {
    for (int j = 0; j < ndim; ++j) {
      F = F_backup;
      F_view(i, j) += del;
      auto energy1 = elast.energy(F_view, {}, {}, {});
      F = F_backup;
      F_view(i, j) -= del;
      auto energy2 = elast.energy(F_view, {}, {}, {});
      EXPECT_NEAR(dE_dF_view(i, j), (energy1 - energy2) / (2 * del), 1e-3);
    }
  }

  // Test the relationship between Hessian and by finite difference.
  Eigen::Matrix<S, ndim * ndim, ndim * ndim> H;
  auto H_view = mp::eigen_support::view(H);
  elast.hessian(H_view, F_view, {}, {}, {});

  auto dE_dF_backup = dE_dF.eval();
  // std::cout << H << std::endl;
  // std::cout << (H - H.transpose()).squaredNorm() << std::endl;

  for (int i = 0; i < ndim; ++i) {
    for (int j = 0; j < ndim; ++j) {
      F = F_backup;
      F_view(i, j) += del;
      elast.stress(dE_dF_view, F_view, {}, {}, {});
      auto dE_dF_forward = dE_dF.eval();
      F = F_backup;
      F_view(i, j) -= del;
      elast.stress(dE_dF_view, F_view, {}, {}, {});
      auto dE_dF_backward = dE_dF.eval();

      auto fwd_view = mp::eigen_support::view(dE_dF_forward);
      auto bwd_view = mp::eigen_support::view(dE_dF_backward);

      for (auto [k, l] : mp::make_shape(ndim, ndim)) {
        S hes_elem = H(i * ndim + j, k * ndim + l);
        S finite_diff = (fwd_view(l, k) - bwd_view(l, k)) / (2 * del);
        EXPECT_NEAR(hes_elem, finite_diff, 1e-3);
      }
    }
  }
}

GTEST_TEST(elast, snh2) {
  using S = double;
  static constexpr index_t ndim = 2;
  using D = mp::device::cpu;
  const S youngs = 1e6, poisson = 0.3;
  const S lambda = elast::compute_lambda(youngs, poisson);
  const S mu = elast::compute_mu(youngs, poisson);
  elast::stable_neohookean<S, D, ndim> elast(lambda, mu);

  auto F = Eigen::Matrix<S, ndim, ndim>::Identity().eval();
  auto F_view = mp::eigen_support::view(F);
  auto dE_dF = Eigen::Matrix<S, ndim, ndim>::Zero().eval();
  auto dE_dF_view = mp::eigen_support::view(dE_dF);

  auto energy = elast.stress(dE_dF_view, F_view, {}, {}, {});
  EXPECT_NEAR(energy, 0.0, 1e-7);
  for (index_t i = 0; i < ndim; ++i) {
    for (index_t j = 0; j < ndim; ++j) {
      EXPECT_NEAR(dE_dF(i, j), 0.0, 1e-7);
    }
  }

  // We test the relationship between energy and stress by finite difference.
  F = Eigen::Matrix<S, ndim, ndim>{{1, 0.3}, {0.2, 3}};
  auto F_backup = F.eval();
  auto energy_center = elast.stress(dE_dF_view, F_view, {}, {}, {});
  auto dE_dF_center = dE_dF.eval();
  std::cout << dE_dF_center << std::endl;

  S del = 1e-6;
  for (int i = 0; i < ndim; ++i) {
    for (int j = 0; j < ndim; ++j) {
      F = F_backup;
      F_view(i, j) += del;
      auto energy1 = elast.energy(F_view, {}, {}, {});
      F = F_backup;
      F_view(i, j) -= del;
      auto energy2 = elast.energy(F_view, {}, {}, {});
      EXPECT_NEAR(dE_dF_view(i, j), (energy1 - energy2) / (2 * del), 1e-3);
    }
  }

  // Test the relationship between Hessian and by finite difference.
  Eigen::Matrix<S, ndim * ndim, ndim * ndim> H;
  auto H_view = mp::eigen_support::view(H);
  elast.hessian(H_view, F_view, {}, {}, {});

  auto dE_dF_backup = dE_dF.eval();

  for (int i = 0; i < ndim; ++i) {
    for (int j = 0; j < ndim; ++j) {
      F = F_backup;
      F_view(i, j) += del;
      elast.stress(dE_dF_view, F_view, {}, {}, {});
      auto dE_dF_forward = dE_dF.eval();
      F = F_backup;
      F_view(i, j) -= del;
      elast.stress(dE_dF_view, F_view, {}, {}, {});
      auto dE_dF_backward = dE_dF.eval();

      for (auto [k, l] : mp::make_shape(ndim, ndim)) {
        S hes_elem = H(i * ndim + j, k * ndim + l);
        S finite_diff = (dE_dF_forward(l, k) - dE_dF_backward(l, k)) / (2 * del);
        S err = std::abs(hes_elem - finite_diff);
        EXPECT_NEAR(err / std::max(std::abs(hes_elem), std::abs(finite_diff)), 0.0, 1e-3);
      }
    }
  }
}