#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/pair.h>

#include <iostream>
#include <mathprim/blas/cpu_blas.hpp>
#include <mathprim/linalg/direct/eigen_support.hpp>  // IWYU pragma: keep
#include <mathprim/linalg/iterative/precond/diagonal.hpp>
#include <mathprim/linalg/iterative/solver/cg.hpp>
#include <mathprim/parallel/openmp.hpp>
#include <mathprim/sparse/blas/naive.hpp>
#include <mathprim/supports/eigen_sparse.hpp>
#include <mathprim/supports/stringify.hpp>

#include "fem.hpp"
#ifdef MATHPRIM_ENABLE_CHOLMOD
#  include "mathprim/linalg/direct/cholmod.hpp"
#endif


#include "ssim/elast/stable_neohookean.hpp"
#include "ssim/finite_elements/stepper/backward_euler.hpp"
#include "ssim/finite_elements/stepper/projective_dynamics.hpp"
#include "ssim/finite_elements/time_step.hpp"
#include "ssim/mesh/common_shapes.hpp"

using Scalar = double;
static constexpr auto csr_f = mathprim::sparse::sparse_format::csr;
using SparseBlas = mp::sparse::blas::naive<Scalar, csr_f>;
using Blas = mp::blas::cpu_blas<Scalar>;
using Par = mp::par::openmp;
using Device = mp::device::cpu;
using StableNeohookean = ssim::elast::stable_neohookean<Scalar, Device, 3>;
using TimeStep = ssim::fem::basic_time_step<Scalar, Device, 3, 4, StableNeohookean, SparseBlas, Blas, Par>;

using Mesh = typename TimeStep::mesh_type;
using MeshView = typename TimeStep::mesh_view;
using ConstMeshView = typename TimeStep::const_mesh_view;
using Hessian = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;

using VertLike = Eigen::Matrix3X<Scalar>;
using Cell = Eigen::Matrix<mp::index_t, 4, Eigen::Dynamic>;

// Linear Sys
#ifdef MATHPRIM_ENABLE_CHOLMOD
using Direct = mp::sparse::direct::cholmod_chol<Scalar, csr_f>;
#else
using Direct = mp::sparse::direct::eigen_simplicial_ldlt<Scalar, csr_f>;
#endif
using Precond = mp::sparse::iterative::diagonal_preconditioner<Scalar, Device, csr_f, Blas>;
using Iterative = mp::sparse::iterative::cg<Scalar, Device, SparseBlas, Blas>;
// Step Solver
using Lbfgs = ssim::fem::time_step_solver_lbfgs;
using LbfgsPd = ssim::fem::time_step_solver_pd<Direct>;
using Newton = ssim::fem::time_step_solver_backward_euler<Iterative>;
using VertNb = nb::ndarray<Scalar, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>;
using CellNb = nb::ndarray<mp::index_t, nb::shape<-1, 4>, nb::device::cpu, nb::c_contig>;
using IdxNb = nb::ndarray<mp::index_t, nb::shape<-1>, nb::device::cpu, nb::c_contig>;

Mesh make_mesh(VertNb vert, CellNb cell) {
  auto vert_view = nbex::to_mp_view_standard(vert);
  auto cell_view = nbex::to_mp_view_standard(cell);

  auto mesh = Mesh(vert_view.shape(0), cell_view.shape(0));
  mp::copy(mesh.vertices(), vert_view);
  mp::copy(mesh.cells(), cell_view);
  return mesh;
}

template <typename Solver>
class timestep_wrapper {
public:
  TimeStep step_;
  Solver solver_;
  bool has_reset_ = false;
  SSIM_INTERNAL_MOVE(timestep_wrapper, default);

  timestep_wrapper(VertNb vert, CellNb cell, Scalar time_step, Scalar youngs_modulus, Scalar poisson_ratio,
                   Scalar density) :
      step_(make_mesh(vert, cell), time_step, youngs_modulus, poisson_ratio, density), solver_() {
  }

  void reset() {
    step_.reset(solver_);
    has_reset_ = true;
  }

  void prepare_step() {
    if (!has_reset_) {
      step_.reset(solver_);
      std::cerr << "Warning: prepare_step() called before reset(), bad logic." << std::endl;
      has_reset_ = true;
    }

    step_.prepare_step();
  }

  void compute_step() { step_.compute_step(solver_); }

  void step_next() { step_.step_next(); }

  double step() {
    auto start = std::chrono::high_resolution_clock::now();
    prepare_step();
    compute_step();
    step_next();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed
        = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    return elapsed.count();
  }

  void mark_dirichlet(mp::index_t node_idx, Eigen::Vector3d targ_deform) {
    auto dofs = step_.dof_type();
    auto targ = step_.dbc_values();

    for (mp::index_t i = 0; i < 3; ++i) {
      dofs(node_idx, i) = ssim::fem::node_boundary_type::dirichlet;
      targ(node_idx, i) = targ_deform[i];
    }
  }

  void mark_dirichlet_batched(Eigen::VectorX<mp::index_t> verts, Eigen::Matrix3Xd targ_deform) {
    auto dofs = step_.dof_type();
    auto targ = step_.dbc_values();

    for (mp::index_t i = 0; i < verts.size(); ++i) {
      for (mp::index_t j = 0; j < 3; ++j) {
        dofs(verts[i], j) = ssim::fem::node_boundary_type::dirichlet;
        targ(verts[i], j) = targ_deform(j, i);
      }
    }
  }

  void set_rtol(Scalar rtol) { step_.set_threshold(rtol); }

  void add_gravity(Eigen::Vector3d gravity) {
    step_.add_ext_force_dof(0, gravity[0]);
    step_.add_ext_force_dof(1, gravity[1]);
    step_.add_ext_force_dof(2, gravity[2]);
  }

  VertLike vertices() const { return mp::eigen_support::cmap(step_.mesh().vertices()); }

  Cell cells() const { return mp::eigen_support::cmap(step_.mesh().cells()); }

  VertLike deformation() const { return mp::eigen_support::cmap(step_.deformation()); }

  VertLike forces() const { return mp::eigen_support::cmap(step_.forces()); }

  Scalar update_energy_and_gradients() { return step_.update_energy_and_gradients(true); }

  void update_hessian(bool make_spsd) { step_.update_hessian(true, make_spsd); }

  Hessian mass_matrix() const noexcept { return mp::eigen_support::map(step_.mass_matrix()); }

  Eigen::VectorX<Scalar> hessian_nonzeros() { return mp::eigen_support::cmap(step_.sysmatrix().values()); }

  Hessian hessian() const { return mp::eigen_support::map(step_.sysmatrix()); }
};

template class timestep_wrapper<Lbfgs>;
template class timestep_wrapper<LbfgsPd>;
template class timestep_wrapper<Newton>;
using namespace nb::literals;
template <typename Solver>
static void bind_ts(nb::module_& m, const char* name) {
  using Wrapped = timestep_wrapper<Solver>;
  nb::class_<Wrapped> step(m, name);
  step.def("reset", &Wrapped::reset, "Setup time step with solver")
      .def("prepare_step", &Wrapped::prepare_step, "Prepare time step")
      .def("compute_step", &Wrapped::compute_step, "Compute time step")
      .def("step_next", &Wrapped::step_next, "Step to next time step")
      .def("step", &Wrapped::step, "Prepare, compute, and step to next time step, return the total solve time.")
      .def("mark_dirichlet", &Wrapped::mark_dirichlet, "Mark dirichlet boundary condition",  //
           "node_idx"_a, "targ_deform"_a)
      .def("mark_dirichlet_batched", &Wrapped::mark_dirichlet_batched,
           "Mark dirichlet boundary condition (batched.)",  //
           "verts"_a, "targ_deform"_a)
      .def("set_rtol", &Wrapped::set_rtol, "Set relative tolerance",  //
           "rtol"_a)
      .def("add_gravity", &Wrapped::add_gravity, "Add gravity")
      .def("vertices", &Wrapped::vertices, "Get vertices")
      .def("cells", &Wrapped::cells, "Get cells")
      .def("deformation", &Wrapped::deformation, "Get deformation")
      .def("forces", &Wrapped::forces, "Get forces")
      .def("update_energy_and_gradients", &Wrapped::update_energy_and_gradients, "Update energy and gradients")
      .def("update_hessian", &Wrapped::update_hessian, "Update hessian",//
           "make_spsd"_a = true)
      .def("mass_matrix", &Wrapped::mass_matrix, "Get mass matrix")
      .def("hessian", &Wrapped::hessian, "Get hessian")
      .def("hessian_nonzeros", &Wrapped::hessian_nonzeros, "Get hessian nonzeros")
      .def(nb::init<VertNb, CellNb, Scalar, Scalar, Scalar, Scalar>(), "Initialize time step",  //
           "vert"_a, "cell"_a, "time_step"_a, "youngs_modulus"_a, "poisson_ratio"_a, "density"_a);
}


///// common shapes /////
std::pair<VertLike, Cell> unit_box(mp::index_t nx, mp::index_t ny, mp::index_t nz) {
  auto mesh = ssim::mesh::unit_box<Scalar>(nx, ny, nz);
  return {mp::eigen_support::cmap(mesh.vertices()), mp::eigen_support::cmap(mesh.cells())};
}

void bind_fem_host(nb::module_& fem_mod) {
  bind_ts<Lbfgs>(fem_mod, "tet_lbfgs");
  bind_ts<LbfgsPd>(fem_mod, "tet_lbfgs_pd");
  bind_ts<Newton>(fem_mod, "tet_newton");
  fem_mod.def("unit_box", &unit_box, "Create a unit box mesh", "nx"_a, "ny"_a, "nz"_a);
}
