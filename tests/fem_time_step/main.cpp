#include <cstdlib>
#include <iostream>

#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/linalg/direct/eigen_support.hpp"
#include "mathprim/parallel/openmp.hpp"
#include "mathprim/sparse/blas/eigen.hpp"
#include "mathprim/supports/stringify.hpp"
#include "ssim/elast/linear.hpp"
#include "ssim/elast/stable_neohookean.hpp"
#include "ssim/finite_elements/backward_euler.hpp"
#include "ssim/finite_elements/time_step.hpp"
#include "ssim/mesh/common_shapes.hpp"

using namespace ssim;
using namespace mathprim;

int main() {
  using Scalar = double;
#ifdef NDEBUG
  index_t nx = 16;  // nx**3 vert.
#else
  index_t nx = 2;
#endif
  auto mesh = mesh::unit_box<Scalar>(nx, nx, nx);
  // auto mesh = mesh::tetra<Scalar>();
  using ElastModel = elast::stable_neohookean<Scalar, device::cpu, 3>;
  using SparseBlas = sparse::blas::eigen<Scalar, sparse::sparse_format::csr>;
  using Blas = blas::cpu_eigen<Scalar>;
  using ParImpl = par::openmp;

  fem::basic_time_step<Scalar, device::cpu, 3, 4, ElastModel, SparseBlas, Blas, ParImpl> step{std::move(mesh), 1e-2,
                                                                                              1e7, 0.33, 1};
  auto dbc = fem::node_boundary_type::dirichlet;
  step.dof_type()(1, 0) = dbc;
  step.dof_type()(1, 1) = dbc;
  step.dof_type()(1, 2) = dbc;
  // step.dof_type()(1, 0) = dbc;
  // step.dof_type()(1, 1) = dbc;
  // step.dof_type()(1, 2) = dbc;

  // External forces.
  using Solver = mp::sparse::direct::eigen_simplicial_ldlt<Scalar, mathprim::sparse::sparse_format::csr>;
  step.add_ext_force_dof(1, -9.8);
  fem::time_step_solver_lbfgs ts_solve;
  // fem::time_step_solver_backward_euler<Solver> ts_solve;
  step.reset(ts_solve);
  Scalar total = 0;
  auto t = step.mass_matrix().visit([&total](index_t /* row */, index_t /* col */, auto& value) {
    total += value;
  });
  par::seq().run(t);
  step.update_hessian();
  // std::cout << eigen_support::map(step.sysmatrix()).toDense() << std::endl;
  std::cout << "Mass: " << total / 3 << std::endl;
  std::cout << "Threshold: " << step.grad_convergence_threshold_abs() << std::endl;
  Solver solver(step.sysmatrix().as_const());
  auto dx_buf = make_buffer<Scalar>(step.forces().shape());

  for (index_t i = 0; i < 2000; ++i) {
    step.prepare_step();
    std::cout << "========== Step " << i << " ==========" << std::endl;
    // std::cout << "Deformation:\n" << eigen_support::cmap(step.deformation()) << std::endl;
    std::cout << "Energy: " << step.update_energy_and_gradients() << std::endl;
    // std::cout << "Force:\n" << eigen_support::cmap(step.forces()) << std::endl;
    // auto force = step.forces();
    // solver.solve(dx.flatten(), force.flatten());
    // std::cout << "Displacement:\n" << eigen_support::cmap(dx) << std::endl;

    Scalar residual = step.compute_step(ts_solve);
    std::cout << "Residual: " << residual << std::endl;
    step.step_next();
  }

  return EXIT_SUCCESS;
}
