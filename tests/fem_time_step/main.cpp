#include <cstdlib>
#include <fstream>
#include <iostream>

#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/linalg/direct/cholmod.hpp"
#include "mathprim/linalg/iterative/precond/diagonal.hpp"
#include "mathprim/linalg/iterative/solver/cg.hpp"
#include "mathprim/parallel/openmp.hpp"
#include "mathprim/sparse/blas/eigen.hpp"
#include "mathprim/supports/stringify.hpp"
#include "ssim/elast/stable_neohookean.hpp"
#include "ssim/finite_elements/stepper/backward_euler.hpp"
#include "ssim/finite_elements/stepper/nonlinear_cg.hpp"
#include "ssim/finite_elements/stepper/projective_dynamics.hpp"
#include "ssim/finite_elements/time_step.hpp"
#include "ssim/mesh/common_shapes.hpp"

using namespace ssim;
using namespace mathprim;

int main(int argc, char** argv) {
  using Scalar = double;
#ifdef NDEBUG
  index_t nx = 6;  // nx**3 vert.
#else
  index_t nx = 6;
#endif

  if (argc > 1) {
    // assume argv[1] is `nx`
    nx = strtol(argv[1], nullptr, 10);
    if (nx < 1) {
      std::cerr << "Invalid nx: " << argv[1] << std::endl;
      return EXIT_FAILURE;
    }
  }

  auto mesh = mesh::unit_box<Scalar>(4 * nx, nx, nx);
  // auto mesh = mesh::tetra<Scalar>();
  using ElastModel = elast::stable_neohookean<Scalar, device::cpu, 3>;
  using SparseBlas = sparse::blas::eigen<Scalar, sparse::sparse_format::csr>;
  using Blas = blas::cpu_eigen<Scalar>;
  using ParImpl = par::openmp;
  mp::functional::affine_transform<Scalar, device::cpu, 3> transform;
  transform.lin_[0] = 4;
  transform.lin_[4] = 1;
  transform.lin_[8] = 1;
  par::seq().vmap(transform, mesh.vertices());

  using Precond = mp::sparse::iterative::diagonal_preconditioner<Scalar, device::cpu,
                                                                 mathprim::sparse::sparse_format::csr, Blas>;
  fem::basic_time_step<Scalar, device::cpu, 3, 4, ElastModel, SparseBlas, Blas, ParImpl> step{
    std::move(mesh), 1e-2, 1e6, 0.4, 10};
  auto dbc = fem::node_boundary_type::dirichlet;
  for (index_t i = 0; i < nx * nx; ++i) {
    step.dof_type()(i, 0) = dbc;
    step.dof_type()(i, 1) = dbc;
    step.dof_type()(i, 2) = dbc;
  }
  // External forces.
  // using Solver = mp::sparse::direct::eigen_simplicial_ldlt<Scalar, mathprim::sparse::sparse_format::csr>;
  using Direct = mp::sparse::direct::cholmod_chol<Scalar, mathprim::sparse::sparse_format::csr>;
  using Iterative = mp::sparse::iterative::cg<Scalar, device::cpu, sparse::blas::eigen<Scalar, sparse::sparse_format::csr>,
                                           blas::cpu_eigen<Scalar>, Precond>;
  // using EigenSolver = Eigen::Sparse
  step.add_ext_force_dof(1, -9.8);
  // fem::time_step_solver_lbfgs ts_solve;
  // fem::time_step_solver_pd<Direct> ts_solve;
  // fem::time_step_solver_backward_euler<Iterative> ts_solve;
  fem::time_step_solver_ncg ts_solve;
  step.set_threshold(1e-3);
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
  auto x_buf = make_buffer<Scalar>(step.forces().shape());
  auto x = x_buf.view();

  index_t total_step = 200;
  auto start = std::chrono::high_resolution_clock::now();
  for (index_t i = 0; i < total_step; ++i) {
    step.prepare_step();
    std::cout << "========== Step " << i << " ==========" << std::endl;
    std::cout << "Energy: " << step.update_energy_and_gradients() << std::endl;
    Scalar residual = step.compute_step(ts_solve);
    std::cout << "Residual: " << residual << std::endl;
    step.step_next();

    auto deform = step.deformation();
    mp::copy(x, deform);
    step.blas().axpy(1.0, step.mesh().vertices(), x);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms"
            << std::endl;
  std::cout << "Average time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / total_step << "ms"
            << std::endl;

  return EXIT_SUCCESS;
}
