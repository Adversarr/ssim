#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>

#include <mathprim/blas/cpu_blas.hpp>
#include <mathprim/parallel/openmp.hpp>
#include <mathprim/sparse/blas/naive.hpp>
#include <mathprim/supports/eigen_sparse.hpp>
#include <memory>

#include "fem.hpp"
#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/linalg/direct/cholmod.hpp"
#include "mathprim/linalg/iterative/precond/diagonal.hpp"
#include "mathprim/linalg/iterative/solver/cg.hpp"
#include "mathprim/parallel/openmp.hpp"
#include "mathprim/sparse/blas/eigen.hpp"
#include "mathprim/supports/stringify.hpp"
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
using Direct = mp::sparse::direct::cholmod_chol<Scalar, csr_f>;
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
  SSIM_INTERNAL_MOVE(timestep_wrapper, default);

  timestep_wrapper(VertNb vert, CellNb cell, Scalar time_step, Scalar youngs_modulus, Scalar poisson_ratio,
                   Scalar density) :
      step_(make_mesh(vert, cell), time_step, youngs_modulus, poisson_ratio, density), solver_() {
    step_.reset(solver_);
  }

  void prepare_step() { step_.prepare_step(); }

  void compute_step() { step_.compute_step(solver_); }

  void step_next() { step_.step_next(); }

  void step() {
    prepare_step();
    compute_step();
    step_next();
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

  void add_gravity(Eigen::Vector3d gravity) {
    step_.add_ext_force_dof(0, gravity[0]);
    step_.add_ext_force_dof(1, gravity[1]);
    step_.add_ext_force_dof(2, gravity[2]);
  }

  VertLike vertices() const { return mp::eigen_support::cmap(step_.mesh().vertices()); }

  Cell cells() const { return mp::eigen_support::cmap(step_.mesh().cells()); }

  VertLike deformation() const { return mp::eigen_support::cmap(step_.deformation()); }

  VertLike forces() const { return mp::eigen_support::cmap(step_.forces()); }

  Hessian hessian() const { return mp::eigen_support::map(step_.sysmatrix()); }
};

template class timestep_wrapper<Lbfgs>;
template class timestep_wrapper<LbfgsPd>;
template class timestep_wrapper<Newton>;

template <typename Solver>
static void bind_ts(nb::module_& m, const char* name) {
  // auto ts = m.("TimeStep", "Time step class");
  nb::class_<timestep_wrapper<Solver>> step(m, name);
  step.def("prepare_step", &timestep_wrapper<Solver>::prepare_step, "Prepare time step")
      .def("compute_step", &timestep_wrapper<Solver>::compute_step, "Compute time step")
      .def("step_next", &timestep_wrapper<Solver>::step_next, "Step to next time step")
      .def("step", &timestep_wrapper<Solver>::step, "Prepare, compute, and step to next time step")
      .def("mark_dirichlet", &timestep_wrapper<Solver>::mark_dirichlet, "Mark dirichlet boundary condition")
      .def("mark_dirichlet_batched", &timestep_wrapper<Solver>::mark_dirichlet_batched,
           "Mark dirichlet boundary condition (batched.)")
      .def("add_gravity", &timestep_wrapper<Solver>::add_gravity, "Add gravity")
      .def("vertices", &timestep_wrapper<Solver>::vertices, "Get vertices")
      .def("cells", &timestep_wrapper<Solver>::cells, "Get cells")
      .def("deformation", &timestep_wrapper<Solver>::deformation, "Get deformation")
      .def("forces", &timestep_wrapper<Solver>::forces, "Get forces")
      .def("hessian", &timestep_wrapper<Solver>::hessian, "Get hessian")
      .def(nb::init<VertNb, CellNb, Scalar, Scalar, Scalar, Scalar>(), "Initialize time step");
}

void bind_fem_host(nb::module_& fem_mod) {
  bind_ts<Lbfgs>(fem_mod, "tet_lbfgs");
  bind_ts<LbfgsPd>(fem_mod, "tet_lbfgs_pd");
  bind_ts<Newton>(fem_mod, "tet_newton");
}