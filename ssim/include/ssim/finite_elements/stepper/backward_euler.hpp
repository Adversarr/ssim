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
    // auto dx_buf = make_buffer<Scalar>(s.forces().shape());
    // auto dx = dx_buf.view();
    // bool converged = false;
    // int iter = 0;
    // while (!converged) {
    //   Scalar f = s.update_energy_and_gradients(false);
    //   converged = s.check_convergence();
    //   if (converged) {
    //     break;
    //   }

    //   std::cout << "Iteration: " << iter++ << " Energy: " << f << std::endl;
    //   s.update_hessian(true, true);
    //   SparseSolver solver(s.sysmatrix().as_const());
    //   solver.solve(dx.flatten(), s.forces().flatten());

    //   auto expected = s.blas().dot(dx, s.forces());
    //   std::cout << "Expected: " << expected << std::endl;
    //   s.blas().axpy(-0.5, dx, s.next_deform());
    // }
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

    std::cout << optimizer.optimize(problem, [](auto info) {
      std::cout << info << std::endl;
    }) << std::endl;;
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
}