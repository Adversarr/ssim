#pragma once
#include <mathprim/optim/optimizer/l_bfgs.hpp>

#include "ssim/finite_elements/time_step.hpp"

namespace ssim::fem {

class time_step_solver_pd : public basic_time_step_solver<time_step_solver_pd> {
public:
  template <typename Scalar, typename Device,                               //
            index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
            typename SparseBlas, typename Blas, typename ParImpl>
  class precond;
  template <typename Scalar, typename Device,                               //
            index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
            typename SparseBlas, typename Blas, typename ParImpl>
  void solve_impl(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>& s) {
    using ls = mathprim::optim::backtracking_linesearcher<Scalar, Device, Blas>;
    using Precond = precond<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>;

    mp::optim::l_bfgs_optimizer<Scalar, Device, Blas, ls, Precond> optimizer;

    variational_problem problem(s);
    problem.setup();
    optimizer.stopping_criteria_.tol_grad_ = s.grad_convergence_threshold_abs();
    optimizer.stopping_criteria_.max_iterations_ = 1000;
    optimizer.optimize(problem);
  }

  template <typename Scalar, typename Device,                               //
            index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
            typename SparseBlas, typename Blas, typename ParImpl>
  void reset_impl(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>&) {}
};
}  // namespace ssim::fem