#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mathprim/blas/cpu_eigen.hpp>

#include "mathprim/blas/cublas.cuh"
#include "mathprim/core/devices/cuda.cuh"
#include "mathprim/linalg/iterative/precond/diagonal.hpp"
#include "mathprim/linalg/iterative/solver/cg.hpp"
#include "mathprim/parallel/cuda.cuh"
#include "mathprim/sparse/blas/cusparse.hpp"
#include "mathprim/supports/io/npy.hpp"
#include "mathprim/supports/stringify.hpp"
#include "ssim/elast/linear.hpp"
#include "ssim/elast/stable_neohookean.hpp"
#include "ssim/finite_elements/stepper/backward_euler.hpp"
#include "ssim/finite_elements/time_step.hpp"
#include "ssim/mesh/common_shapes.hpp"

using namespace ssim;
using namespace mathprim;

int main() {
  using Scalar = double;
#ifdef NDEBUG
  index_t nx = 20;  // nx**3 vert.
#else
  index_t nx = 6;
#endif
  auto mesh = mesh::unit_box<Scalar>(4 * nx, nx, nx);
  // auto mesh = mesh::tetra<Scalar>();
  using ElastModel = elast::stable_neohookean<Scalar, device::cuda, 3>;
  using SparseBlas = sparse::blas::cusparse<Scalar, mathprim::sparse::sparse_format::csr>;
  using Blas = blas::cublas<Scalar>;
  using ParImpl = par::cuda;
  mp::functional::affine_transform<Scalar, device::cpu, 3> transform;
  transform.lin_[0] = 4;
  transform.lin_[4] = 1;
  transform.lin_[8] = 1;
  par::seq().vmap(transform, mesh.vertices());


  fem::basic_time_step<Scalar, device::cuda, 3, 4, ElastModel, SparseBlas, Blas, ParImpl> step{mesh.to(device::cuda()),
                                                                                               1e-2, 1e6, 0.4, 10};
  step.parallel().run(nx * nx, [dof_type = step.dof_type()] __device__(index_t i) {
    dof_type(i, 0) = fem::node_boundary_type::dirichlet;
    dof_type(i, 1) = fem::node_boundary_type::dirichlet;
    dof_type(i, 2) = fem::node_boundary_type::dirichlet;
  });

  // External forces.
  using Precond = mp::sparse::iterative::diagonal_preconditioner<Scalar, device::cuda,
                                                                 mathprim::sparse::sparse_format::csr, Blas>;
  using Solver = mp::sparse::iterative::cg<Scalar, device::cuda, SparseBlas, Blas, Precond>;


  step.add_ext_force_dof(1, -9.8);
  // fem::time_step_solver_lbfgs ts_solve;
  fem::time_step_solver_backward_euler<Solver> ts_solve;
  step.set_threshold(1e-3);
  step.reset(ts_solve);
  Scalar total = 0;
  auto t = step.mass_matrix().visit([&total](index_t /* row */, index_t /* col */, auto& value) {
    total += value;
  });
  par::seq().run(t);
  step.update_hessian();
  // std::cout << eigen_support::map(step.sysmatrix()).toDense() << std::endl;
  // std::cout << "Mass: " << total / 3 << std::endl;
  // std::cout << "Threshold: " << step.grad_convergence_threshold_abs() << std::endl;
  // Solver solver(step.sysmatrix().as_const());
  auto x_buf = make_buffer<Scalar>(step.forces().shape());
  auto x = x_buf.view();
  auto d_buf = make_buffer<Scalar>(step.forces().shape());
  auto d = d_buf.view();

  index_t total_step = 200;
  auto start = std::chrono::high_resolution_clock::now();
  for (index_t i = 0; i < total_step; ++i) {
    step.prepare_step();
    std::cout << "========== Step " << i << " ==========" << std::endl;
    std::cout << "Energy: " << step.update_energy_and_gradients() << std::endl;
    Scalar residual = step.compute_step(ts_solve);
    std::cout << "Residual: " << residual << std::endl;
    step.step_next();
    // // save the result.
    // mp::io::numpy<Scalar, 2> writer;
    // copy(x, step.deformation());
    // blas::cpu_eigen<double> eigen;
    // eigen.axpy(1.0, mesh.vertices(), x);
    // std::ofstream out_file("deform_" + std::to_string(i) + ".npy", std::ios::binary);
    // writer.write(out_file, x);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms"
            << std::endl;
  std::cout << "Average time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / total_step << "ms"
            << std::endl;

  return EXIT_SUCCESS;
}
