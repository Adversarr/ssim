#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/pair.h>

#include <iostream>
#include <mathprim/blas/cublas.cuh>
#include <mathprim/core/defines.hpp>
#include <mathprim/linalg/iterative/precond/diagonal.hpp>
#include <mathprim/linalg/iterative/solver/cg.hpp>
#include <mathprim/parallel/cuda.cuh>
#include <mathprim/sparse/blas/cusparse.hpp>
#include <mathprim/supports/eigen_dense.hpp>
#include <mathprim/supports/eigen_sparse.hpp>
#include <mathprim/supports/stringify.hpp>

#include "fem.hpp"
#include "ssim/elast/stable_neohookean.hpp"
#include "ssim/finite_elements/stepper/backward_euler.hpp"
#include "ssim/finite_elements/stepper/nonlinear_cg.hpp"
#include "ssim/finite_elements/time_step.hpp"
#include "ssim/mesh/common_shapes.hpp"

using Scalar = double;
static constexpr auto csr_f = mathprim::sparse::sparse_format::csr;
using SparseBlas = mp::sparse::blas::cusparse<Scalar, csr_f>;
using Blas = mp::blas::cublas<Scalar>;
using Par = mp::par::cuda;
using Device = mp::device::cuda;
using StableNeohookean = ssim::elast::stable_neohookean<Scalar, Device, 3>;
using TimeStep = ssim::fem::basic_time_step<Scalar, Device, 3, 4, StableNeohookean, SparseBlas, Blas, Par>;

using Mesh = typename TimeStep::mesh_type;
using MeshView = typename TimeStep::mesh_view;
using ConstMeshView = typename TimeStep::const_mesh_view;
using Hessian = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;

using VertLike = Eigen::Matrix3X<Scalar>;
using Cell = Eigen::Matrix<mp::index_t, 4, Eigen::Dynamic>;

// Linear Sys
using Precond = mp::sparse::iterative::diagonal_preconditioner<Scalar, Device, csr_f, Blas>;
using Iterative = mp::sparse::iterative::cg<Scalar, Device, SparseBlas, Blas>;
// Step Solver
using Lbfgs = ssim::fem::time_step_solver_lbfgs;
using Newton = ssim::fem::time_step_solver_backward_euler<Iterative>;
using NonlinearCgExt = ssim::fem::time_step_solver_ncg_with_ext_prec<Scalar, Device, SparseBlas>;
using VertNb = nb::ndarray<Scalar, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>;
using CellNb = nb::ndarray<mp::index_t, nb::shape<-1, 4>, nb::device::cpu, nb::c_contig>;
using IdxNb = nb::ndarray<mp::index_t, nb::shape<-1>, nb::device::cpu, nb::c_contig>;

static Mesh make_mesh(VertNb vert, CellNb cell) {
  auto vert_view = nbex::to_mathprim(vert);
  auto cell_view = nbex::to_mathprim(cell);

  auto mesh = Mesh(vert_view.shape(0), cell_view.shape(0));
  mp::copy(mesh.vertices(), vert_view);
  mp::copy(mesh.cells(), cell_view);
  return mesh;
}

template <typename Solver>
class timestep_wrapper_cuda {
public:
  TimeStep step_;
  Solver solver_;
  bool has_reset_ = false;
  SSIM_INTERNAL_MOVE(timestep_wrapper_cuda, default);

  timestep_wrapper_cuda(VertNb vert, CellNb cell, Scalar time_step, Scalar youngs_modulus, Scalar poisson_ratio,
                        Scalar density) :
      step_(make_mesh(vert, cell), time_step, youngs_modulus, poisson_ratio, density), solver_() {}

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

  void mark_dirichlet_batched(Eigen::VectorX<mp::index_t> verts, Eigen::Matrix3Xd targ_deform) {
    auto dofs = step_.dof_type();
    auto targ = step_.dbc_values();
    auto& parallel = step_.parallel();

    auto n_verts = verts.size();
    auto h_verts = mp::eigen_support::view(verts);
    auto h_targ_deform = mp::eigen_support::view(targ_deform);

    auto d_vert_buf = mp::make_buffer<mp::index_t, Device>(h_verts.shape());
    auto d_deform_buf = mp::make_buffer<double, Device>(h_targ_deform.shape());
    auto d_verts = d_vert_buf.view();
    auto d_deform = d_deform_buf.view();
    copy(d_verts, h_verts);
    copy(d_deform, h_targ_deform);

    parallel.run(n_verts, [d_verts, d_deform, dofs, targ] __device__(mp::index_t idx) {
      auto vert = d_verts[idx];
      for (mp::index_t j = 0; j < 3; ++j) {
        dofs(vert, j) = ssim::fem::node_boundary_type::dirichlet;
        targ(vert, j) = d_deform(idx, j);
      }
    });
  }

  void mark_general_batched(Eigen::VectorX<mp::index_t> verts) {
    auto dofs = step_.dof_type();
    auto targ = step_.dbc_values();
    auto& parallel = step_.parallel();

    auto n_verts = verts.size();
    auto h_verts = mp::eigen_support::view(verts);
    auto d_vert_buf = mp::make_buffer<mp::index_t, Device>(h_verts.shape());
    auto d_verts = d_vert_buf.view();
    copy(d_verts, h_verts);

    parallel.run(n_verts, [d_verts, dofs, targ] __device__(mp::index_t idx) {
      auto vert = d_verts[idx];
      for (mp::index_t j = 0; j < 3; ++j) {
        dofs(vert, j) = ssim::fem::node_boundary_type::general;
        targ(vert, j) = 0.0;  // Set to zero or any default value for general constraints
      }
    });
  }

  void set_rtol(Scalar rtol) { step_.set_threshold(rtol); }

  void add_gravity(Eigen::Vector3d gravity) {
    step_.add_ext_force_dof(0, gravity[0]);
    step_.add_ext_force_dof(1, gravity[1]);
    step_.add_ext_force_dof(2, gravity[2]);
  }

  VertLike deformation() const {
    const mp::index_t n_vert = step_.mesh().num_vertices();
    VertLike deform(3, n_vert);
    mp::copy(mp::eigen_support::view(deform), step_.deformation());
    return deform;
  }

  VertLike forces() const {
    const mp::index_t n_vert = step_.mesh().num_vertices();
    VertLike forces(3, n_vert);
    mp::copy(mp::eigen_support::view(forces), step_.forces());
    return forces;
  }
};

using namespace mp;
class timestep_ncg : public timestep_wrapper_cuda<NonlinearCgExt> {
public:
  using timestep_wrapper_cuda<NonlinearCgExt>::timestep_wrapper_cuda;
  using matrix = sparse::basic_sparse_matrix<Scalar, Device, csr_f>;

  void reset() {
    step_.reset(solver_);
    has_reset_ = true;
    auto sysmatrix = step_.sysmatrix();
    index_t n = sysmatrix.rows(), nnz = sysmatrix.nnz();
    matrix_ = matrix(n, n, nnz);
    copy(matrix_.values().view(), sysmatrix.values());
    copy(matrix_.outer_ptrs().view(), sysmatrix.outer_ptrs());
    copy(matrix_.inner_indices().view(), sysmatrix.inner_indices());

    // init with identity
    step_.parallel().run(n,                                          //
                         [vals = matrix_.values().view(),            //
                          outer_ptrs = matrix_.outer_ptrs().view(),  //
                          inner_indices = matrix_.inner_indices().view()] __device__(index_t row) {
                           index_t row_start = outer_ptrs[row], row_end = outer_ptrs[row + 1];
                           for (index_t i = row_start; i < row_end; ++i) {
                             index_t col = inner_indices[i];
                             if (row == col) {
                               vals[i] = 1;
                             } else {
                               vals[i] = 0;
                             }
                           }
                         });
  }

  void set_matrix(Eigen::VectorXd values) { copy(matrix_.values().view(), eigen_support::view(values)); }

  sparse::basic_sparse_matrix<Scalar, Device, csr_f> matrix_;
};

using namespace nb::literals;
template <typename SolverWrapped>
static void bind_solver(nb::module_& m, const char* name) {
  nb::class_<SolverWrapped> c(m, name);
  c.def(nb::init<VertNb, CellNb, Scalar, Scalar, Scalar, Scalar>())
      .def("step", &SolverWrapped::step)
      .def("reset", &SolverWrapped::reset)
      .def("prepare_step", &SolverWrapped::prepare_step)
      .def("compute_step", &SolverWrapped::compute_step)
      .def("step_next", &SolverWrapped::step_next)
      .def("mark_dirichlet_batched", &SolverWrapped::mark_dirichlet_batched)
      .def("mark_general_batched", &SolverWrapped::mark_general_batched)
      .def("set_rtol", &SolverWrapped::set_rtol)
      .def("add_gravity", &SolverWrapped::add_gravity)
      .def("deformation", &SolverWrapped::deformation);
}

static void bind_ncg_ext(nb::module_& m) {
  using SolverWrapped = timestep_ncg;
  nb::class_<SolverWrapped> c(m, "tet_ncg_cuda_ext_ai");
  c.def(nb::init<VertNb, CellNb, Scalar, Scalar, Scalar, Scalar>())
      .def("step", &SolverWrapped::step)
      .def("reset", &SolverWrapped::reset)
      .def("prepare_step", &SolverWrapped::prepare_step)
      .def("compute_step", &SolverWrapped::compute_step)
      .def("step_next", &SolverWrapped::step_next)
      .def("mark_dirichlet_batched", &SolverWrapped::mark_dirichlet_batched,  //
           "idxs"_a, "deform"_a)
      .def("mark_general_batched", &SolverWrapped::mark_general_batched,  //
           "idxs"_a)
      .def("set_rtol", &SolverWrapped::set_rtol)
      .def("add_gravity", &SolverWrapped::add_gravity)
      .def("deformation", &SolverWrapped::deformation)
      .def("set_matrix", &SolverWrapped::set_matrix,  //
           "values"_a.noconvert());
}

void bind_fem_cuda(nb::module_& fem_mod) {
  bind_solver<timestep_wrapper_cuda<Lbfgs>>(fem_mod, "tet_lbfgs_cuda");
  bind_solver<timestep_wrapper_cuda<Newton>>(fem_mod, "tet_newton_cuda");
  bind_ncg_ext(fem_mod);
}
