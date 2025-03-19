#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>

#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/sparse/blas/eigen.hpp"
#include "ssim/elast/linear.hpp"
#include "ssim/elast/stable_neohookean.hpp"
#include "mathprim/linalg/direct/eigen_support.hpp"
#include "ssim/finite_elements/time_step.hpp"
#include "ssim/mesh/common_shapes.hpp"
#include "mathprim/supports/stringify.hpp"

using namespace ssim;
using namespace mathprim;

GTEST_TEST(diff, tetra) {
  using Scalar = double;
  // auto mesh = mesh::unit_box<Scalar>(2, 2, 2);
  auto mesh = mesh::tetra<Scalar>();
  using ElastModel = elast::linear<Scalar, device::cpu, 3>;
  using SparseBlas = sparse::blas::eigen<Scalar, sparse::sparse_format::csr>;
  using Blas = blas::cpu_eigen<Scalar>;
  using ParImpl = par::seq;
  Scalar dt = 1e-2;
  fem::basic_time_step<Scalar, device::cpu, 3, 4, ElastModel, SparseBlas, Blas, ParImpl> step{std::move(mesh), dt};
  // auto dbc = fem::node_boundary_type::dirichlet;
  // step.dof_type()(0, 0) = dbc;
  // step.dof_type()(0, 1) = dbc;
  // step.dof_type()(0, 2) = dbc;

  auto deform = eigen_support::cmap(step.next_deform());
  deform(0, 0) = -0.1;
  deform(1, 0) = -0.1;
  deform(2, 0) = -0.1;
  deform(0, 1) = 0.1;
  deform(1, 1) = -0.1;
  deform(2, 1) = -0.1;
  deform(0, 2) = -0.1;
  deform(1, 2) = 0.1;
  deform(2, 2) = -0.1;
  deform(0, 3) = 0.1;
  deform(1, 3) = 0.1;
  deform(2, 3) = -0.1;
  auto backup = deform.eval();
  eigen_support::cmap(step.inertia_deform().flatten()).setRandom();
  step.update_energy_and_gradients();

  // Test if finite diff of energy is gradient.
  Scalar delta = 1e-3;
  auto grad = eigen_support::cmap(step.forces()).eval();
  auto finite_diff_grad = grad.eval();
  for (index_t vi = 0; vi < 4; ++vi) {
    for (index_t dof = 0; dof < 3; ++dof) {
      deform = backup;
      deform(dof, vi) += delta;
      auto energy_forward = step.update_energy_and_gradients();
      deform = backup;
      deform(dof, vi) -= delta;
      auto energy_backward = step.update_energy_and_gradients();
      finite_diff_grad(dof, vi) = (energy_forward - energy_backward) / (2 * delta);
    }
  }
  EXPECT_NEAR((finite_diff_grad - grad).squaredNorm(), 0.0, 1e-4);

  deform = backup;
  step.update_energy_and_gradients();
  step.update_hessian();
  auto hessian = eigen_support::map(step.sysmatrix()).toDense().eval();
  auto hessian_fd = hessian.eval();

  for (index_t vi = 0; vi < 4; ++vi) {
    for (index_t dof = 0; dof < 3; ++dof) {
      // If gradient is correct, directly derive it from gradient.
      deform = backup;
      deform(dof, vi) += delta;
      step.update_energy_and_gradients();
      auto grad_forward = eigen_support::cmap(step.forces().flatten()).eval();
      deform = backup;
      deform(dof, vi) -= delta;
      step.update_energy_and_gradients();
      auto grad_backward = eigen_support::cmap(step.forces().flatten()).eval();
      Eigen::VectorX<Scalar> grad_diff = (grad_forward - grad_backward) / (2 * delta);
      hessian_fd.col(vi * 3 + dof) = grad_diff;
    }
  }
  std::cout << hessian.topLeftCorner(6, 6) << std::endl;
  std::cout << hessian_fd.topLeftCorner(6, 6) << std::endl;
  EXPECT_NEAR((hessian_fd - hessian).squaredNorm(), 0.0, 1e-4);
}


GTEST_TEST(diff, unit_box) {
  using Scalar = double;
  // auto mesh = mesh::unit_box<Scalar>(2, 2, 2);
  auto mesh = mesh::unit_box<Scalar>();
  using ElastModel = elast::linear<Scalar, device::cpu, 3>;
  using SparseBlas = sparse::blas::eigen<Scalar, sparse::sparse_format::csr>;
  using Blas = blas::cpu_eigen<Scalar>;
  using ParImpl = par::seq;
  Scalar dt = 1e-2;
  fem::basic_time_step<Scalar, device::cpu, 3, 4, ElastModel, SparseBlas, Blas, ParImpl> step{std::move(mesh), dt};
  // auto dbc = fem::node_boundary_type::dirichlet;
  // step.dof_type()(0, 0) = dbc;
  // step.dof_type()(0, 1) = dbc;
  // step.dof_type()(0, 2) = dbc;
  auto deform = eigen_support::cmap(step.next_deform());
  step.prepare_step();

  auto backup = (Eigen::MatrixX<Scalar>::Random(deform.rows(), deform.cols()) * 3e-1).eval();
  deform = backup;
  // auto backup = deform.Zero(deform.rows(), deform.cols()).eval();
  // eigen_support::cmap(step.inertia_deform().flatten()).setRandom();

  Scalar energy0 = step.update_energy_and_gradients();
  std::cout << "Energy: " << energy0 << std::endl;
  // Test if finite diff of energy is gradient.
  Scalar delta = 1e-3;
  auto grad = eigen_support::cmap(step.forces()).eval();
  auto finite_diff_grad = grad.eval();
  std::cout << grad << std::endl;
  for (index_t vi = 0; vi < step.mesh().num_vertices(); ++vi) {
    for (index_t dof = 0; dof < 3; ++dof) {
      deform = backup;
      deform(dof, vi) += delta;
      auto energy_forward = step.update_energy_and_gradients();
      deform = backup;
      deform(dof, vi) -= delta;
      auto energy_backward = step.update_energy_and_gradients();
      finite_diff_grad(dof, vi) = (energy_forward - energy_backward) / (2 * delta);
    }
  }
  EXPECT_NEAR((finite_diff_grad - grad).squaredNorm(), 0.0, 1e-4);



  deform = backup;
  step.update_energy_and_gradients();
  step.update_hessian();
  auto hessian = eigen_support::map(step.sysmatrix()).toDense().eval();
  auto hessian_fd = hessian.eval();

  for (index_t vi = 0; vi < step.mesh().num_vertices(); ++vi) {
    for (index_t dof = 0; dof < 3; ++dof) {
      // If gradient is correct, directly derive it from gradient.
      deform = backup;
      deform(dof, vi) += delta;
      step.update_energy_and_gradients();
      auto grad_forward = eigen_support::cmap(step.forces().flatten()).eval();
      deform = backup;
      deform(dof, vi) -= delta;
      step.update_energy_and_gradients();
      auto grad_backward = eigen_support::cmap(step.forces().flatten()).eval();
      Eigen::VectorX<Scalar> grad_diff = (grad_forward - grad_backward) / (2 * delta);
      hessian_fd.col(vi * 3 + dof) = grad_diff;
    }
  }
  std::cout << hessian.topLeftCorner(6, 6) << std::endl;
  std::cout << hessian_fd.topLeftCorner(6, 6) << std::endl;
  EXPECT_NEAR((hessian_fd - hessian).squaredNorm(), 0.0, 1e-4);
}
