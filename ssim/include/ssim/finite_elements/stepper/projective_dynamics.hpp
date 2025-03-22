#pragma once
#include <mathprim/linalg/basic_sparse_solver.hpp>
#include <mathprim/optim/optimizer/l_bfgs.hpp>

#include "ssim/finite_elements/time_step.hpp"

namespace ssim::fem {

template <typename LinSolver>
class time_step_solver_pd : public basic_time_step_solver<time_step_solver_pd<LinSolver>> {
public:
  using Scalar = typename LinSolver::scalar_type;
  using Device = typename LinSolver::device_type;
  using pd_matrix_type = mp::sparse::basic_sparse_matrix<Scalar, Device, mp::sparse::sparse_format::csr>;
  class precond
      : public mp::optim::l_bfgs_preconditioner<time_step_solver_pd::precond, Scalar, Device> {
  public:
    using base = mp::optim::l_bfgs_preconditioner<time_step_solver_pd::precond, Scalar, Device>;
    LinSolver* solver_;
    precond() = default;
    MATHPRIM_INTERNAL_MOVE(precond, default);
    void set_solver(LinSolver* solver) { solver_ = solver; }
    using vector_type = typename base::vector_type;
    using const_vector = typename base::const_vector;

    mp::sparse::basic_sparse_solver<LinSolver, Scalar, Device, mp::sparse::sparse_format::csr>& solver() {
      assert(solver_ && "Solver is not set.");
      return *solver_;
    }

    void apply_impl(vector_type z, const_vector q, const_vector /* s */, const_vector /* y */) {
      // mp::copy(z, q);  // z <- q
      solver().solve(z, q);
    }
  };
  template <index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
            typename SparseBlas, typename Blas, typename ParImpl>
  void solve_impl(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>& s) {
    using ls = mathprim::optim::backtracking_linesearcher<Scalar, Device, Blas>;
    mp::optim::l_bfgs_optimizer<Scalar, Device, Blas, ls, precond> optimizer;
    optimizer.preconditioner_.set_solver(&solver_);
    // s.update_hessian(true, true);
    // solver_.compute(s.sysmatrix().as_const());
    variational_problem problem(s);
    problem.setup();
    optimizer.stopping_criteria_.tol_grad_ = s.grad_convergence_threshold_abs();
    optimizer.stopping_criteria_.max_iterations_ = 1000;
    optimizer.optimize(problem);
  }

  template <index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
            typename SparseBlas, typename Blas, typename ParImpl>
  void reset_impl(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>& s) {
    auto v = s.mesh().vertices();
    auto e = s.mesh().cells();
    // computes laplacian + mass
    mp::par::seq pf;
    basic_unstructured<Scalar, mp::device::cpu, PhysicalDim, TopologyDim> mesh(s.mesh().num_vertices(),
                                                                               s.mesh().num_cells());
    mp::copy(mesh.vertices(), v);
    mp::copy(mesh.cells(), e);
    auto mass = mass_integrator(mesh.view(), s.density());
    auto dt = s.time_step(), approx_diff = s.uniform_elasticity().approximated_diffusivity();
    auto laplace = laplace_integrator(mesh.view(), (dt * dt * approx_diff));
    auto local_buffer = mp::make_buffer<Scalar>(mesh.num_cells(), mp::holder<TopologyDim>{}, mp::holder<TopologyDim>{});
    local_buffer.fill_bytes(0);
    auto local = local_buffer.view();
    pf.run(mass, local);
    pf.run(laplace, local);

    std::vector<mp::sparse::entry<Scalar>> entries;
    entries.reserve(mesh.num_cells() * TopologyDim * TopologyDim * PhysicalDim);
    for (index_t elem = 0; elem < mesh.num_cells(); ++elem) {
      for (auto [i, j] : mp::make_shape(TopologyDim, TopologyDim)) {
        Scalar val = local(elem, i, j);
        for (index_t dof = 0; dof < PhysicalDim; ++dof) {
          const index_t row = mesh.cells()(elem, i) * s.dofs_per_node + dof;
          const index_t col = mesh.cells()(elem, j) * s.dofs_per_node + dof;
          entries.emplace_back(row, col, val);
        }
      }
    }

    auto coo = mp::sparse::make_from_triplets<Scalar>(entries.begin(), entries.end(), s.total_dofs(), s.total_dofs(),
                                                      mp::sparse::sparse_property::symmetric);
    auto csr = mp::sparse::make_from_coos<Scalar, mp::sparse::sparse_format::csr>(coo);
    pd_matrix_ = csr.to(Device{});

    // Filter DBC.
    auto enforcer = s.boundary_enforcer();
    enforcer.hessian(pf, pd_matrix_.view());
    solver_.compute(pd_matrix_.const_view());
  }

  LinSolver solver_;
  pd_matrix_type pd_matrix_;
};


template <typename LinSolver>
class time_step_solver_h0 : public basic_time_step_solver<time_step_solver_h0<LinSolver>> {
public:
  using Scalar = typename LinSolver::scalar_type;
  using Device = typename LinSolver::device_type;
  using pd_matrix_type = mp::sparse::basic_sparse_matrix<Scalar, Device, mp::sparse::sparse_format::csr>;
  class precond
      : public mp::optim::l_bfgs_preconditioner<time_step_solver_h0::precond, Scalar, Device> {
  public:
    using base = mp::optim::l_bfgs_preconditioner<time_step_solver_h0::precond, Scalar, Device>;
    LinSolver* solver_;
    precond() = default;
    MATHPRIM_INTERNAL_MOVE(precond, default);
    void set_solver(LinSolver* solver) { solver_ = solver; }
    using vector_type = typename base::vector_type;
    using const_vector = typename base::const_vector;

    mp::sparse::basic_sparse_solver<LinSolver, Scalar, Device, mp::sparse::sparse_format::csr>& solver() {
      assert(solver_ && "Solver is not set.");
      return *solver_;
    }

    void apply_impl(vector_type z, const_vector q, const_vector /* s */, const_vector /* y */) {
      // mp::copy(z, q);  // z <- q
      solver().solve(z, q);
    }
  };
  template <index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
            typename SparseBlas, typename Blas, typename ParImpl>
  void solve_impl(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>& s) {
    using ls = mathprim::optim::backtracking_linesearcher<Scalar, Device, Blas>;
    mp::optim::l_bfgs_optimizer<Scalar, Device, Blas, ls, precond> optimizer;
    optimizer.preconditioner_.set_solver(&solver_);
    s.update_hessian(true, true);
    solver_.factorize();
    variational_problem problem(s);
    problem.setup();
    optimizer.stopping_criteria_.tol_grad_ = s.grad_convergence_threshold_abs();
    optimizer.stopping_criteria_.max_iterations_ = 1000;
    optimizer.optimize(problem);
    // std::cout << optimizer.optimize(problem, [](const auto& res) {
    //   std::cout << res << std::endl;
    // }) << std::endl;
  }

  template <index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
            typename SparseBlas, typename Blas, typename ParImpl>
  void reset_impl(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>& s) {
    solver_.analyze(s.sysmatrix().as_const());
  }

  LinSolver solver_;
};

}  // namespace ssim::fem
