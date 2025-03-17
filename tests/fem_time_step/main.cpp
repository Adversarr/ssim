#include <cstdlib>
#include <iostream>

#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/sparse/blas/eigen.hpp"
#include "ssim/elast/linear.hpp"
#include "ssim/finite_elements/boundary.hpp"
#include "ssim/finite_elements/time_step.hpp"
#include "ssim/mesh/common_shapes.hpp"
#include "mathprim/supports/stringify.hpp"

using namespace ssim;
using namespace mathprim;

int main() {
  auto mesh = mesh::unit_box<float>(2, 2, 2);
  auto nE = mesh.num_cells();
  using ElastModel = elast::linear<float, device::cpu, 3>;
  using SparseBlas = sparse::blas::eigen<float, sparse::sparse_format::csr>;
  using Blas = blas::cpu_eigen<float>;
  using ParImpl = par::seq;
  fem::basic_time_step<float, device::cpu, 3, 4, ElastModel, SparseBlas, Blas, ParImpl> step{std::move(mesh)};
  auto dbc = fem::node_boundary_type::dirichlet;
  step.dof_type()(0, 0) = dbc;
  step.dof_type()(0, 1) = dbc;
  step.dof_type()(0, 2) = dbc;
  step.reset();

  // External forces.
  step.add_ext_force_dof(1, -9.8);
  std::cout << step.sysmatrix() << std::endl;
  float total = 0;
  auto t = step.sysmatrix().visit([&total](index_t /* row */, index_t /* col */, float value) {
    total += value;
  });
  par::seq().run(t);
  std::cout << "Mass: " << total / 3 << std::endl;

  for (index_t i = 0; i < 10; ++i) {
    step.prepare_step();
    std::cout << "========== Step " << i << " ==========" << std::endl;
    std::cout << "Deformation:\n" << eigen_support::cmap(step.deformation()) << std::endl;
    std::cout << "Energy: " << step.update_energy_and_gradients() << std::endl;
    step.step_next();
  }

  return EXIT_SUCCESS;
}
