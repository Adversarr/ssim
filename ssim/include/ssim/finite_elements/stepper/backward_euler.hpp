#pragma once
#include <mathprim/optim/optimizer/l_bfgs.hpp>
#include <mathprim/optim/optimizer/newton.hpp>

#include "ssim/finite_elements/time_step.hpp"

namespace ssim::fem {

template <typename SparseSolver>
class time_step_solver_backward_euler : public basic_time_step_solver<time_step_solver_backward_euler<SparseSolver>> {
public:
  template <typename Scalar, typename Device,                               //
  index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
  typename SparseBlas, typename Blas, typename ParImpl>
  void solve_impl(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>& s) {
    variational_problem problem(s);
    problem.setup();
    using linesearch = mp::optim::backtracking_linesearcher<Scalar, Device, Blas>;
    mp::optim::newton_optimizer<Scalar, Device, Blas, linesearch, SparseSolver> optimizer;
    optimizer.stopping_criteria_.tol_grad_ = s.grad_convergence_threshold_abs();
    optimizer.stopping_criteria_.max_iterations_ = 1000;
    optimizer.linesearcher().armijo_threshold_ = 0.3;
    optimizer.set_hessian_fn([&]() {
      return problem.eval_hessian_impl();
    });

    auto result_s = optimizer.optimize(problem);
    result.converged_ = result_s.converged_;
    result.iterations_ = result_s.iterations_;
    result.grad_norm_ = result_s.grad_norm_;
    result.last_change_ = result_s.last_change_;
    result.value_ = result_s.value_;
  }

  template <typename Scalar, typename Device,                               //
            index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
            typename SparseBlas, typename Blas, typename ParImpl>
  void reset_impl(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>&) {}

  mp::optim::optim_result<double> result;
};


class time_step_solver_lbfgs : public basic_time_step_solver<time_step_solver_lbfgs> {
public:
  template <typename Scalar, typename Device,                               //
  index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
  typename SparseBlas, typename Blas, typename ParImpl>
  void solve_impl(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>& s) {
    mp::optim::l_bfgs_optimizer<Scalar, Device, Blas> optimizer;
    variational_problem problem(s);
    problem.setup();
    optimizer.stopping_criteria_.tol_grad_ = s.grad_convergence_threshold_abs();
    optimizer.stopping_criteria_.max_iterations_ = 1000;
    
    auto result_s = optimizer.optimize(problem);
    result.converged_ = result_s.converged_;
    result.iterations_ = result_s.iterations_;
    result.grad_norm_ = result_s.grad_norm_;
    result.last_change_ = result_s.last_change_;
    result.value_ = result_s.value_;
  }

  template <typename Scalar, typename Device,                               //
            index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
            typename SparseBlas, typename Blas, typename ParImpl>
  void reset_impl(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>&) {}
  mp::optim::optim_result<double> result;
};
}