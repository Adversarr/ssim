#pragma once
#include <mathprim/optim/optimizer/l_bfgs.hpp>

#include "time_step.hpp"

namespace ssim::fem {
template <typename SparseSolver>
class time_step_solver_backward_euler : public basic_time_step_solver<time_step_solver_backward_euler<SparseSolver>> {
public:
  template <typename Scalar, typename Device,                               //
  index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
  typename SparseBlas, typename Blas, typename ParImpl>
  void solve_impl(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>& s) {
    auto dx_buf = make_buffer<Scalar>(s.forces().shape());
    auto dx = dx_buf.view();
    bool converged = false;
    int iter = 0;
    while (!converged) {
      Scalar f = s.update_energy_and_gradients(false);
      converged = s.check_convergence();
      if (converged) {
        break;
      }

      std::cout << "Iteration: " << iter++ << " Energy: " << f << std::endl;
      s.update_hessian(true, true);
      SparseSolver solver(s.sysmatrix().as_const());
      solver.solve(dx.flatten(), s.forces().flatten());

      auto expected = s.blas().dot(dx, s.forces());
      std::cout << "Expected: " << expected << std::endl;
      s.blas().axpy(-0.5, dx, s.next_deform());
    }
  }

  template <typename Scalar, typename Device,                               //
            index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
            typename SparseBlas, typename Blas, typename ParImpl>
  void reset_impl(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>&) {}
};


class time_step_solver_lbfgs : public basic_time_step_solver<time_step_solver_lbfgs> {
public:
  template <typename Scalar, typename Device,                               //
  index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
  typename SparseBlas, typename Blas, typename ParImpl>
  void solve_impl(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>& s) {
    mp::optim::l_bfgs_optimizer<Scalar, Device, Blas> optimizer;
    class variational_problem: public mp::optim::basic_problem<variational_problem, Scalar, Device> {
      using base = mp::optim::basic_problem<variational_problem, Scalar, Device>;
      friend base;
    public:
      using timestep_type = basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>;
      timestep_type& step_;
      explicit variational_problem(timestep_type& step): step_(step) {
        base::register_parameter(step_.next_deform().flatten());
      }

      ~variational_problem() { std::cout << "Total Evaluations: " << cnt_ << std::endl; }

    protected:
      void eval_value_and_gradients_impl() {
        auto dst = base::at(0).gradient();
        base::accumulate_loss(step_.update_energy_and_gradients(false));
        mp::copy(dst, step_.forces().flatten());
        ++cnt_;
      }

      void eval_value_impl() { eval_value_and_gradients_impl(); }
      void eval_gradients_impl() { eval_value_and_gradients_impl(); }

      index_t cnt_ = 0;
    };

    variational_problem problem(s);
    problem.setup();
    optimizer.stopping_criteria_.tol_grad_ = s.grad_convergence_threshold_abs();
    optimizer.stopping_criteria_.max_iterations_ = 1000;
    optimizer.optimize(problem, [](auto result) {
      std::cout << "Iter[" << result.iterations_ << "]: loss=" << result.value_ << ", |g|=" << result.grad_norm_
                << std::endl;
    });
  }

  template <typename Scalar, typename Device,                               //
            index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
            typename SparseBlas, typename Blas, typename ParImpl>
  void reset_impl(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>&) {}
};
}