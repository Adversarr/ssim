#pragma once
#include <iostream>
#include <mathprim/optim/optimizer/ncg.hpp>

#include "ssim/finite_elements/time_step.hpp"

namespace ssim::fem {

class time_step_solver_ncg : public basic_time_step_solver<time_step_solver_ncg> {
public:
  template <typename Scalar, typename Device,                               //
            index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
            typename SparseBlas, typename Blas, typename ParImpl>
  void solve_impl(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>& s) {
    variational_problem problem(s);
    problem.setup();
    using linesearch = mp::optim::backtracking_linesearcher<Scalar, Device, Blas>;
    mp::optim::ncg_optimizer<Scalar, Device, Blas, linesearch> optimizer;
    optimizer.stopping_criteria_.tol_grad_ = s.grad_convergence_threshold_abs();
    optimizer.stopping_criteria_.max_iterations_ = 1000;
    optimizer.learning_rate_ = 1e-2;
    optimizer.strategy_ = mp::optim::ncg_strategy::dai_kou;

    auto result_s = optimizer.optimize(problem);
    result.converged_ = result_s.converged_;
    result.iterations_ = result_s.iterations_;
    result.grad_norm_ = result_s.grad_norm_;
    result.last_change_ = result_s.last_change_;
    result.value_ = result_s.value_;
    std::cout << result_s << std::endl;
  }

  template <typename Scalar, typename Device,                               //
            index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
            typename SparseBlas, typename Blas, typename ParImpl>
  void reset_impl(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>&) {}

  mp::optim::optim_result<double> result;
};

template <typename Scalar, typename Device, typename SparseBlas>
class time_step_solver_ncg_with_ext_prec : public basic_time_step_solver<time_step_solver_ncg> {
public:
  template <index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
            typename Blas, typename ParImpl>
  void solve_impl(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>& s) {
    struct ainv_precond : public mp::optim::ncg_preconditioner<ainv_precond, Scalar, Device> {
      using base = mp::optim::ncg_preconditioner<ainv_precond, Scalar, Device>;
      using vector_type = typename base::vector_type;
      using const_vector = typename base::const_vector;
      ainv_precond() = default;

      void apply_impl(vector_type x, const_vector g) {
        // z = lo.T * x.
        bl_->gemv(1, g, 0, z, true);
        // y = lo * y.
        bl_->gemv(1, z, 0, x, false);
      }

      vector_type z;
      SparseBlas* bl_;
    };

    variational_problem problem(s);
    problem.setup();
    using linesearch = mp::optim::backtracking_linesearcher<Scalar, Device, Blas>;
    mp::optim::ncg_optimizer<Scalar, Device, Blas, linesearch, ainv_precond> optimizer;
    optimizer.stopping_criteria_.tol_grad_ = s.grad_convergence_threshold_abs();
    optimizer.stopping_criteria_.max_iterations_ = 1000;
    optimizer.learning_rate_ = 1e-2;
    optimizer.strategy_ = mp::optim::ncg_strategy::dai_kou;
    optimizer.preconditioner_.bl_ = &sparse_;
    optimizer.preconditioner_.z = z_.view();

    auto result_s = optimizer.optimize(problem);
    result.converged_ = result_s.converged_;
    result.iterations_ = result_s.iterations_;
    result.grad_norm_ = result_s.grad_norm_;
    result.last_change_ = result_s.last_change_;
    result.value_ = result_s.value_;
    std::cout << result_s << std::endl;
  }

  template <index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
            typename Blas, typename ParImpl>
  void reset_impl(
      basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>& step) {
    index_t total_dofs = step.mesh().num_vertices() * PhysicalDim;
    z_ = mp::make_buffer<Scalar>(total_dofs);
    z_.fill_bytes(0);
  }

  mp::optim::optim_result<double> result;
  SparseBlas sparse_;
  mp::contiguous_vector_buffer<Scalar, Device> z_;
};

}  // namespace ssim::fem
