#pragma once
#include <iostream>
#include <mathprim/blas/blas.hpp>
#include <mathprim/core/buffer.hpp>
#include <mathprim/sparse/basic_sparse.hpp>
#include <mathprim/sparse/cvt.hpp>
#include <mathprim/sparse/gather.hpp>
#include <stdexcept>

#include "boundary.hpp"
#include "ssim/defines.hpp"
#include "ssim/elast/basic_elast.hpp"
#include "ssim/finite_elements/bilinears.hpp"
#include "ssim/finite_elements/def_grad.hpp"
#include "ssim/finite_elements/global_composer.hpp"
#include "ssim/finite_elements/rest_vol.hpp"
#include "ssim/mesh/basic_mesh.hpp"

namespace ssim::fem {

namespace internal {

template <typename Scalar, typename Device, index_t PhysicalDim, index_t TopologyDim, typename HessianOp>
struct wrapped_pfpx {
  static constexpr index_t dofs_per_node = PhysicalDim;
  static constexpr index_t hes_nrows = TopologyDim * dofs_per_node;
  static constexpr index_t elem_hes_nrows = PhysicalDim * PhysicalDim;
  using const_pfpx_type = mp::contiguous_view<const Scalar, mp::shape_t<elem_hes_nrows, hes_nrows>, Device>;
  using element_hessian = mp::contiguous_view<Scalar, mp::shape_t<elem_hes_nrows, elem_hes_nrows>, Device>;
  using vert_hessian = mp::contiguous_view<Scalar, mp::shape_t<hes_nrows, hes_nrows>, Device>;
  using const_def_grad = mp::contiguous_view<const Scalar, mp::shape_t<PhysicalDim, PhysicalDim>, Device>;

  HessianOp op_;
  mp::batched<const_pfpx_type> pfpx_;
  bool make_spsd_ = false;

  SSIM_PRIMFUNC wrapped_pfpx(HessianOp op, mp::batched<const_pfpx_type> pfpx, bool make_spsd) :
      op_(op), pfpx_(pfpx), make_spsd_(make_spsd) {}
  SSIM_INTERNAL_ENABLE_ALL_CTOR(wrapped_pfpx);

  SSIM_PRIMFUNC void operator()(vert_hessian dst, const_def_grad F, const_pfpx_type pfpx) const noexcept {
    // note: the shape dim in comment assumes 3d applications.
    using local_mat = Eigen::Matrix<Scalar, elem_hes_nrows, elem_hes_nrows>;
    local_mat temp_buf;  // [9, 9]
    auto temp = mp::eigen_support::view(temp_buf);
    op_(temp, F);
    auto pfpx_eigen = mp::eigen_support::cmap(pfpx);  // [12, 9]
    auto dst_eigen = mp::eigen_support::cmap(dst);    // [12, 12]
    if (make_spsd_) {
      Eigen::SelfAdjointEigenSolver<local_mat> solve_eigen(temp_buf);
      auto eigvals = solve_eigen.eigenvalues().cwiseMax(0).eval();
      auto eigvecs = solve_eigen.eigenvectors().eval();

      temp_buf = eigvecs * eigvals.asDiagonal() * eigvecs.transpose();
    }

    dst_eigen = pfpx_eigen * temp_buf * pfpx_eigen.transpose();
  }
};

template <typename Scalar, typename Device, index_t PhysicalDim, index_t TopologyDim>
struct stress_element_to_vertex {
  using element_stress = mp::contiguous_view<Scalar, mp::shape_t<PhysicalDim, PhysicalDim>, Device>;
  using vert_stress = mp::contiguous_view<Scalar, mp::shape_t<TopologyDim, PhysicalDim>, Device>;

  using dminv_item = mp::contiguous_view<const Scalar, mp::shape_t<PhysicalDim, PhysicalDim>, Device>;
  using dminv_type = mp::batched<dminv_item>;

  SSIM_PRIMFUNC stress_element_to_vertex() = default;
  SSIM_INTERNAL_ENABLE_ALL_CTOR(stress_element_to_vertex);

  SSIM_PRIMFUNC void operator()(vert_stress dst, element_stress src, dminv_item dminv) const noexcept {
    // dst: [3, 4], src: [3, 3], dminv[3, 3]
    // dst[1...4] = src * dminv.T
    auto src_eigen = mp::eigen_support::cmap(src).transpose();
    auto dminv_eigen = mp::eigen_support::cmap(dminv);
    Eigen::Matrix<Scalar, PhysicalDim, PhysicalDim> out = src_eigen * dminv_eigen;
    // std::cout << out << std::endl;
    for (index_t j = 0; j < PhysicalDim; ++j) {
      Scalar dof_total = 0;
      for (index_t i = 1; i < TopologyDim; ++i) {
        const Scalar val = out(j, i - 1);
        dst(i, j) = val;
        dof_total += val;
      }
      dst(0, j) = -dof_total;
    }
  }
};

template <typename Scalar, typename Device, index_t PhysicalDim, index_t TopologyDim, typename HessianOp>
auto make_wrapped_pfpx(
    HessianOp op,
    mp::contiguous_view<const Scalar, mp::shape_t<mp::keep_dim, PhysicalDim * PhysicalDim, PhysicalDim * TopologyDim>,
                        Device>
        pfpx,
    bool make_spsd) {
  return wrapped_pfpx<Scalar, Device, PhysicalDim, TopologyDim, HessianOp>(op, pfpx, make_spsd);
}
}  // namespace internal

template <typename Scalar, typename Device,                               //
          index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
          typename SparseBlas, typename Blas, typename ParImpl>
class basic_time_step;

template <typename Derived>
class basic_time_step_solver {
public:
  using derived_type = Derived;

  SSIM_PRIMFUNC Derived& derived() noexcept { return static_cast<Derived&>(*this); }
  SSIM_PRIMFUNC const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }

  template <typename Scalar, typename Device,                               //
            index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
            typename SparseBlas, typename Blas, typename ParImpl>
  void solve(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>& step) {
    derived().solve_impl(step);
  }

  template <typename Scalar, typename Device,                               //
            index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
            typename SparseBlas, typename Blas, typename ParImpl>
  void reset(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>& step) {
    derived().reset_impl(step);
  }
};

class time_step_solver_nothing : public basic_time_step_solver<time_step_solver_nothing> {
public:
  template <typename Scalar, typename Device,                               //
            index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
            typename SparseBlas, typename Blas, typename ParImpl>
  void solve_impl(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>&) {}

  template <typename Scalar, typename Device,                               //
            index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
            typename SparseBlas, typename Blas, typename ParImpl>
  void reset_impl(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>&) {}
};

/**
 * Mesh stage: (in constructor, not editable)
 *    -> mesh topology, determine matrix, buffer size.
 * Precompute stage:
 *    -> setup youngs/poisson/density
 *    -> setup rest pose information for deformation gradient.
 *    -> setup previous, current deformation, velocity
 *    -> setup acceleration solvers, e.g. Projective Dynamics
 *    -> mark dirichlet boundaries.
 * Loop stage:
 *    -> prepare_step: update dbc values, extermal acceleration
 *    -> compute_step: put result into next_deform
 *    -> step_next: step all state buffers.
 * ------------
 * Some helper functions are available after precompute stage.
 *    - update_deformation_gradient: update deformation gradient
 *    - update_energy: update incremental potential's value
 *    - update_gradients: update incremental potential's value, and its gradient
 *                        the operation is fused.
 *    - update_hessian: update its hessian matrix
 */

// The variational form is :
//    argmin 1/2 |u - u_inertia|_M^2  + dt^2 E(u)
template <typename Scalar, typename Device,                               //
          index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
          typename SparseBlas, typename Blas, typename ParImpl>
class basic_time_step {
public:
  using scalar_type = Scalar;
  using device_type = Device;
  static constexpr index_t physical_dim = PhysicalDim;
  static constexpr index_t topology_dim = TopologyDim;
  static constexpr index_t dofs_per_node = PhysicalDim;
  static constexpr index_t hessian_nrows = topology_dim * dofs_per_node;
  using blas_ref = mp::blas::basic_blas<Blas, Scalar, Device>&;
  using const_blas_ref = const mp::blas::basic_blas<Blas, Scalar, Device>&;
  using par_ref = mp::par::parfor<ParImpl>&;
  using const_par_ref = const mp::par::parfor<ParImpl>&;

  // Each term, can have its own local stiffness matrix, force vector, energy value.
  using local_stiffness = mp::contiguous_view<Scalar, mp::shape_t<hessian_nrows, hessian_nrows>, Device>;
  using const_local_matrix = mp::contiguous_view<const Scalar, mp::shape_t<hessian_nrows, hessian_nrows>, Device>;
  using local_force = mp::contiguous_view<Scalar, mp::shape_t<topology_dim, dofs_per_node>, Device>;
  using const_local_force = mp::contiguous_view<const Scalar, mp::shape_t<topology_dim, dofs_per_node>, Device>;
  using local_stress = mp::contiguous_view<Scalar, mp::shape_t<dofs_per_node, dofs_per_node>, Device>;

  using batched_local_stiffness = mp::batched<local_stiffness>;
  using batched_local_force = mp::batched<local_force>;
  using batched_local_stress = mp::batched<local_stress>;
  using batched_local_energy = mp::contiguous_vector_view<Scalar, Device>;

  using local_stiffness_buffer = mp::to_buffer_t<batched_local_stiffness>;
  using local_force_buffer = mp::to_buffer_t<batched_local_force>;
  using local_stress_buffer = mp::to_buffer_t<batched_local_stress>;
  using local_energy_buffer = mp::to_buffer_t<batched_local_energy>;

  using const_batched_local_stiffness = mp::batched<const_local_matrix>;
  using const_batched_local_force = mp::batched<const_local_force>;
  using const_batched_local_energy = mp::contiguous_vector_view<const Scalar, Device>;

  using sys_matrix = sparse_matrix<Scalar, Device>;
  using sys_matrix_view = sparse_view<Scalar, Device>;
  using const_sys_matrix_view = sparse_view<const Scalar, Device>;

  // Mesh
  using mesh_type = basic_unstructured<Scalar, Device, PhysicalDim, TopologyDim>;
  using host_mesh_type = basic_unstructured<Scalar, mp::device::cpu, PhysicalDim, TopologyDim>;
  using boundary_type = boundary_condition<Scalar, Device, PhysicalDim, TopologyDim, dofs_per_node>;
  using dof_type_type = mp::contiguous_view<node_boundary_type, mp::shape_t<PhysicalDim>, Device>;
  using const_dof_type = mp::contiguous_view<const node_boundary_type, mp::shape_t<PhysicalDim>, Device>;
  using batched_dof_type = mp::batched<dof_type_type>;
  using const_batched_dof_type = mp::batched<const_dof_type>;
  using batched_dof_type_buffer = mp::to_buffer_t<batched_dof_type>;

  using mesh_view = basic_unstructured_view<Scalar, Device, PhysicalDim, TopologyDim>;
  using vertex_type = typename mesh_type::vertex_type;
  using cell_type = typename mesh_type::cell_type;
  using batched_vertex = typename mesh_type::batched_vertex;
  using batched_cell = typename mesh_type::batched_cell;

  using const_mesh_view = basic_unstructured_view<const Scalar, Device, PhysicalDim, TopologyDim>;
  using const_vertex = typename const_mesh_view::vertex_type;
  using const_cell = typename const_mesh_view::cell_type;
  using const_batched_vertex = typename const_mesh_view::batched_vertex;
  using const_batched_cell = typename const_mesh_view::batched_cell;

  using vertex_buffer = typename mesh_type::batched_vertex_buffer;
  using cell_buffer = typename mesh_type::batched_cell_buffer;

  // Elast
  using rest_volume = mp::contiguous_vector_view<Scalar, Device>;
  using const_rest_volume = mp::contiguous_vector_view<const Scalar, Device>;
  using rest_volume_buffer = mp::to_buffer_t<rest_volume>;
  using deform_grad_item = mp::contiguous_view<Scalar, mp::shape_t<PhysicalDim, PhysicalDim>, Device>;
  using deform_grad_type = mp::batched<deform_grad_item>;
  using deform_grad_buffer = mp::to_buffer_t<deform_grad_type>;
  using dminv_item = mp::contiguous_view<Scalar, mp::shape_t<PhysicalDim, PhysicalDim>, Device>;
  using dminv_type = mp::batched<dminv_item>;
  using dminv_buffer = mp::to_buffer_t<dminv_type>;
  using pfpx_item = mp::contiguous_view<Scalar, mp::shape_t<PhysicalDim * PhysicalDim, hessian_nrows>, Device>;
  using pfpx_type = mp::batched<pfpx_item>;
  using pfpx_buffer = mp::to_buffer_t<pfpx_type>;
  using def_grad = deformation_gradient<Scalar, Device, physical_dim, topology_dim>;

  struct dof_mod_work {
    index_t dof_idx_;
    Scalar x_;
    SSIM_PRIMFUNC void operator()(vertex_type accel) const noexcept { accel[dof_idx_] += x_; }
  };

  ////////////////////////////////////////////////
  /// Construct stage
  ////////////////////////////////////////////////
  SSIM_INTERNAL_ENABLE_ALL_CTOR(basic_time_step);
  explicit basic_time_step(mesh_type mesh,           //
                           Scalar time_step = 1e-3,  //
                           Scalar youngs = 1e6, Scalar poisson = 0.33, Scalar density = 1e3) :
      mesh_(std::move(mesh)), youngs_(youngs), poisson_(poisson), density_(density), time_step_(time_step) {
    reset();
  }

  blas_ref blas() noexcept { return blas_; }
  const_blas_ref blas() const noexcept { return blas_; }
  par_ref parallel() noexcept { return pf_; }
  const_par_ref parallel() const noexcept { return pf_; }

  ////////////////////////////////////////////////
  /// Getter/Setters
  ////////////////////////////////////////////////
  mesh_view mesh() noexcept { return mesh_.view(); }
  const_mesh_view mesh() const noexcept { return mesh_.view(); }
  batched_dof_type dof_type() noexcept { return dof_type_.view(); }
  const_batched_dof_type dof_type() const noexcept { return dof_type_.view(); }
  batched_vertex dbc_values() noexcept { return dbc_values_.view(); }
  const_batched_vertex dbc_values() const noexcept { return dbc_values_.view(); }
  batched_vertex deformation() noexcept { return deformation_.view(); }
  const_batched_vertex deformation() const noexcept { return deformation_.view(); }
  batched_vertex prev_deform() noexcept { return prev_deform_.view(); }
  const_batched_vertex prev_deform() const noexcept { return prev_deform_.view(); }
  batched_vertex inertia_deform() noexcept { return inertia_deform_.view(); }
  const_batched_vertex inertia_deform() const noexcept { return inertia_deform_.view(); }
  batched_vertex next_deform() noexcept { return next_deform_.view(); }
  const_batched_vertex next_deform() const noexcept { return next_deform_.view(); }
  batched_vertex velocity() noexcept { return velocity_.view(); }
  const_batched_vertex velocity() const noexcept { return velocity_.view(); }
  batched_vertex ext_accel() noexcept { return ext_accel_.view(); }
  const_batched_vertex ext_accel() const noexcept { return ext_accel_.view(); }
  batched_vertex forces() noexcept { return forces_.view(); }
  const_batched_vertex forces() const noexcept { return forces_.view(); }

  sys_matrix_view sysmatrix() noexcept { return sysmatrix_.view(); }
  const_sys_matrix_view sysmatrix() const noexcept { return sysmatrix_.view(); }

  sys_matrix_view mass_matrix() noexcept { return mass_matrix_.view(); }
  const_sys_matrix_view mass_matrix() const noexcept { return mass_matrix_.view(); }

  sys_matrix_view mass_matrix_filtered() noexcept { return mass_matrix_filtered_.view(); }
  const_sys_matrix_view mass_matrix_filtered() const noexcept { return mass_matrix_filtered_.view(); }

  Scalar grad_convergence_threshold() const noexcept { return gradient_convergence_threshold_; }
  Scalar grad_convergence_threshold_abs() const noexcept { return gradient_convergence_threshold_abs_; }

  basic_time_step& set_time_step(Scalar time_step) {
    SSIM_INTERNAL_CHECK_THROW(time_step > 0, std::invalid_argument, "Time step must be positive.");
    time_step_ = time_step;
    return *this;
  }
  basic_time_step& set_youngs(Scalar youngs) {
    SSIM_INTERNAL_CHECK_THROW(youngs > 0, std::invalid_argument, "Young's modulus must be positive.");
    youngs_ = youngs;
    return *this;
  }
  basic_time_step& set_poisson(Scalar poisson) {
    SSIM_INTERNAL_CHECK_THROW(poisson > 0 && poisson < 0.5, std::invalid_argument,
                              "Poisson's ratio must be in (0, 0.5).");
    poisson_ = poisson;
    return *this;
  }
  basic_time_step& set_density(Scalar density) {
    SSIM_INTERNAL_CHECK_THROW(density > 0, std::invalid_argument, "Density must be positive.");
    density_ = density;
    return *this;
  }
  basic_time_step& set_damping(Scalar damping) {
    SSIM_INTERNAL_CHECK_THROW(0 <= damping && damping < 1, std::invalid_argument, "Damping must be in [0, 1).");
    damping_ = damping;
    return *this;
  }

  basic_time_step& set_threshold(Scalar threshold) {
    SSIM_INTERNAL_CHECK_THROW((0 < threshold) && (threshold < 1), std::invalid_argument, "Threshold must be positive.");
    gradient_convergence_threshold_ = threshold;
    return *this;
  }

  Scalar time_step() const noexcept { return time_step_; }
  Scalar youngs() const noexcept { return youngs_; }
  Scalar poisson() const noexcept { return poisson_; }
  Scalar density() const noexcept { return density_; }
  Scalar damping() const noexcept { return damping_; }

  ////////////////////////////////////////////////
  /// Precompute stage helpers
  ////////////////////////////////////////////////
  void reset() {
    time_step_solver_nothing alg;
    reset(alg);
  }
  template <typename Algorithm = time_step_solver_nothing>
  void reset(basic_time_step_solver<Algorithm>& solver) {
    SSIM_INTERNAL_CHECK_THROW(youngs_ > 0, std::invalid_argument, "Young's modulus must be positive.");
    SSIM_INTERNAL_CHECK_THROW(poisson_ > 0 && poisson_ < 0.5, std::invalid_argument,
                              "Poisson's ratio must be in (0, 0.5).");
    SSIM_INTERNAL_CHECK_THROW(density_ > 0, std::invalid_argument, "Density must be positive.");
    using mp::make_buffer;
    host_mesh_ = mesh_.to(mp::device::cpu{});
    const index_t n_elem = mesh_.num_cells(), n_vert = mesh_.num_vertices();

    if (0.5 - poisson_ < 1e-4) {
      const double poisson = poisson_;
      fprintf(stderr, "Warning: got extremely high Poisson's ratio, may cause numerical instability. %lf\n", poisson);
    }

    {  /// Basic
      auto zero_buffer = [this]() {
        auto buf = make_buffer<Scalar, Device>(mesh_.vertices().shape());
        buf.fill_bytes(0);
        return buf;
      };
      deformation_ = zero_buffer();
      inertia_deform_ = zero_buffer();
      prev_deform_ = zero_buffer();
      next_deform_ = zero_buffer();
      velocity_ = zero_buffer();
      forces_ = zero_buffer();
      temp_buffer_ = zero_buffer();
    }

    {
      if (!ext_accel_) {
        ext_accel_ = make_buffer<Scalar, Device>(mesh_.vertices().shape());
        ext_accel_.fill_bytes(0);
      }
      if (!dbc_values_) {
        dbc_values_ = make_buffer<Scalar, Device>(mesh_.vertices().shape());
        dbc_values_.fill_bytes(0);
      }
      if (!dof_type_) {
        dof_type_ = make_buffer<node_boundary_type, Device>(mesh_.vertices().shape());
        dof_type_.fill_bytes(0);
      }
    }

    {  /// Elasticity
      rest_volume_ = make_buffer<Scalar, Device>(n_elem);
      dminv_ = make_buffer<Scalar, Device>(n_elem, mp::holder<PhysicalDim>{}, mp::holder<PhysicalDim>{});
      pfpx_ = make_buffer<Scalar, Device>(n_elem, mp::holder<PhysicalDim * PhysicalDim>{}, mp::holder<hessian_nrows>{});
      deform_grad_ = make_buffer<Scalar, Device>(n_elem, mp::holder<PhysicalDim>{}, mp::holder<PhysicalDim>{});
      local_energy_ = make_buffer<Scalar, Device>(n_elem);
      local_force_ = make_buffer<Scalar, Device>(n_elem, mp::holder<topology_dim>{}, mp::holder<dofs_per_node>{});
      local_stress_ = make_buffer<Scalar, Device>(n_elem, mp::holder<dofs_per_node>{}, mp::holder<dofs_per_node>{});
      local_stiffness_ = make_buffer<Scalar, Device>(n_elem, mp::holder<hessian_nrows>{}, mp::holder<hessian_nrows>{});
    }

    {  /// Deformation gradients
      def_grad dg(mesh_.const_view());
      parallel().run(dg.compute_dminv(dminv_.view()));
      parallel().run(dg.compute_pfpx(pfpx_.view(), dminv_.view()));
      parallel().run(make_rest_vol_task(mesh_.view(), rest_volume_.view()));
    }

    auto h_cells = host_mesh_.cells();
    std::vector<mp::sparse::entry<Scalar>> all_entries;
    all_entries.reserve(n_elem * hessian_nrows * hessian_nrows);

    {  // Mass
      auto integrator = mass_integrator(host_mesh_.view(), density_);
      auto local_mass_buf = mp::make_buffer<Scalar>(n_elem, topology_dim, topology_dim);
      local_mass_buf.fill_bytes(0);
      auto local_mass = local_mass_buf.view();
      mp::par::seq().run(integrator, local_mass);

      for (index_t elem = 0; elem < n_elem; ++elem) {
        for (auto [i, j] : mp::make_shape(topology_dim, topology_dim)) {
          for (auto [i_dof, j_dof] : mp::make_shape(dofs_per_node, dofs_per_node)) {
            const index_t row = h_cells(elem, i) * dofs_per_node + i_dof;
            const index_t col = h_cells(elem, j) * dofs_per_node + j_dof;
            Scalar val = i_dof == j_dof ? local_mass(elem, i, j) : 0;
            all_entries.emplace_back(row, col, val);
          }
        }
      }
    }
    index_t total_dofs = n_vert * dofs_per_node;
    auto coo = mp::sparse::make_from_triplets<Scalar>(all_entries.begin(), all_entries.end(), total_dofs, total_dofs,
                                                      mp::sparse::sparse_property::symmetric);
    auto csr = mp::sparse::make_from_coos<Scalar, mp::sparse::sparse_format::csr>(coo);
    mass_matrix_ = csr.to(Device{});
    mass_matrix_filtered_ = csr.to(Device{});
    sysmatrix_ = csr.to(Device{});

    // Filter DBC.
    auto vals = dbc_values_.const_view();
    boundary_type enforce(mesh_.const_view(), dof_type(), vals);
    enforce.hessian(parallel(), mass_matrix_filtered_.view());
    // std::cout << blas().asum(mass_matrix_.values().view()) << std::endl;
    // std::cout << blas().asum(mass_matrix_filtered_.values().view()) << std::endl;

    mass_bl_ = SparseBlas(mass_matrix_.const_view());
    sys_bl_ = SparseBlas(sysmatrix_.const_view());
    mass_filtered_bl_ = SparseBlas(mass_matrix_filtered_.const_view());

    {  // Gather Info
      local_global_composer<Scalar, Device, PhysicalDim, TopologyDim, PhysicalDim> composer;
      auto mesh = mesh_.const_view();
      auto rest_volume = rest_volume_.const_view();
      stress_gather_info_ = composer.force(mesh, rest_volume);
      hessian_gather_info_ = composer.hessian(mesh, rest_volume);
    }

    {  // convergence criteria
      auto tmp = temp_buffer_.view();
      auto tmp2 = inertia_deform_.view();  // fuse.
      temp_buffer_.fill_bytes(0);
      inertia_deform_.fill_bytes(0);
      parallel().vmap(dof_mod_work(0, time_step_ * time_step_), tmp);
      mass_bl_.gemv(1.0, tmp.flatten(), 0.0, tmp2.flatten());
      Scalar body_force = blas().norm(tmp2);
      gradient_convergence_threshold_abs_ = body_force * gradient_convergence_threshold_;
    }
    solver.reset(*this);
  }

  ////////////////////////////////////////////////
  /// Loop Stage
  ////////////////////////////////////////////////
  void prepare_step() {
    SSIM_INTERNAL_CHECK_THROW(0 <= damping_ && damping_ < 1, std::invalid_argument,
                              "velocity damping must be in [0, 1).");
    auto inertia = inertia_deform_.view();
    auto deform = deformation_.view();
    auto vel = velocity_.view();
    auto accel = ext_accel_.view();
    auto solution = next_deform_.view();
    // 1. u_Predict <- u + dt * v + 0.5 * dt^2 * a
    mp::copy(inertia, deform);
    blas().axpy(time_step_, vel, inertia);
    blas().axpy(0.5 * time_step_ * time_step_, accel, inertia);

    // 2. apply dirichlet boundary conditions.
    auto vals = dbc_values_.const_view();
    boundary_type enforce(mesh_.const_view(), dof_type(), vals);
    enforce.value(parallel(), inertia);

    mp::copy(solution, inertia);  // initialize solution to our prediction.
  }

  template <typename Algorithm>
  Scalar compute_step(basic_time_step_solver<Algorithm>& stepper) {
    stepper.solve(*this);

    // check convergence
    update_energy_and_gradients(false);
    auto force = forces_.view();
    return blas().norm(force);
  }

  void step_next() {
    auto next = next_deform_.view();
    auto prev = prev_deform_.view();
    auto current = deformation_.view();
    auto vel = velocity_.view();
    mp::copy(prev, current);
    mp::copy(current, next);
    mp::copy(vel, current);
    blas().axpy(-1.0, prev, vel);
    blas().scal(Scalar(1.0) / time_step_, vel);

    // damp the velocity.
    blas().scal(1 - damping_, vel);
  }

  ////////////////////////////////////////////////
  /// Helpers
  ////////////////////////////////////////////////

  /// @brief add external force to a specific dof (uniformly.)
  void add_ext_force_dof(index_t dof_idx, Scalar del) {
    SSIM_INTERNAL_CHECK_THROW(-PhysicalDim < dof_idx && dof_idx < PhysicalDim, std::invalid_argument,
                              "|dof_idx| must be less than PhysicalDim.");
    if (dof_idx < 0) {
      dof_idx += PhysicalDim;
    }
    parallel().vmap(dof_mod_work{dof_idx, del}, ext_accel_);
  }

  void update_deformation_gradient() {
    def_grad dg(mesh_.const_view());
    auto deform_grad = deform_grad_.view();
    parallel().run(dg.compute_def_grad(deform_grad, dminv_.view(), next_deform()));
  }

  Scalar update_energy_and_gradients(bool is_deform_grad_updated = false) {
    if (!is_deform_grad_updated) {
      update_deformation_gradient();
    }
    forces_.fill_bytes(0);
    auto f = forces_.view();

    /// Inertia Part: 1/2 ||X - X_inertia||_MassFiltered^2
    /// => grad = Mass(X - X_inertia)
    auto inertia = inertia_deform_.view();
    auto deform = next_deform_.view();
    auto temp = temp_buffer_.view();
    blas().copy(temp, deform);
    blas().axpy(-1.0, inertia.flatten(), temp.flatten());  // temp = X - X_inertia
    mass_bl_.gemv(1.0, temp.flatten(), 0.0, f.flatten());  // f <- M(X - X_inertia)
    Scalar inertia_energy = 0.5 * blas().dot(temp.flatten(), f.flatten());

    /// Elasticity Part: dt^2 grad_X [Energy(F(X))]
    // TODO: svd.
    auto lambda = elast::compute_lambda(youngs_, poisson_);
    auto mu = elast::compute_mu(youngs_, poisson_);
    ElastModel model(lambda, mu);
    auto local_energy = local_energy_.view();
    auto local_force = local_force_.view();
    auto local_stress = local_stress_.view();
    auto deform_grad = deform_grad_.view();
    parallel().vmap(mp::par::make_output_vmapped(model.stress_op()), local_energy, local_stress, deform_grad);
    internal::stress_element_to_vertex<Scalar, Device, physical_dim, topology_dim> to_vert;
    parallel().vmap(to_vert, local_force, local_stress, dminv_.const_view());

    auto n_elem = mesh_.num_cells();
    mp::sparse::basic_gather_operator<Scalar, Device, 1> gather_op{
      f,                                                          // force on vertices,
      local_force.reshape(n_elem * topology_dim, dofs_per_node),  // force on elements
      stress_gather_info_.desc(),                                 // cached gather information
      time_step_ * time_step_};                                   // alpha = dt^2
    parallel().run(gather_op);

    // filter out the dirichlet boundary conditions.
    boundary_type enforce(mesh_.const_view(), dof_type(), dbc_values());
    enforce.grads(parallel(), f);

    // Compute energy
    auto rest_volume = rest_volume_.const_view();
    Scalar elast_energy = blas().dot(local_energy, rest_volume)  * time_step_ * time_step_;
    // std::cout << "Inertia: " << inertia_energy << " Elasticity: " << elast_energy << std::endl;
    return inertia_energy + elast_energy;
  }

  void update_hessian(bool is_deform_grad_updated = false, bool make_spsd = false) {
    if (!is_deform_grad_updated) {
      update_deformation_gradient();
    }

    mp::zeros(local_stiffness_.view());
    auto sysmat_nonzeros = sysmatrix_.values().view();
    mp::copy(sysmat_nonzeros, mass_matrix_.values().view());

    // TODO: svd.
    auto lambda = elast::compute_lambda(youngs_, poisson_);
    auto mu = elast::compute_mu(youngs_, poisson_);
    ElastModel model(lambda, mu);
    auto local_stiffness = local_stiffness_.view();
    auto deform_grad = deform_grad_.view();
    auto compute_hessian = model.hessian_op();
    auto pfpx = pfpx_.const_view();
    auto wrapped_pfpx
        = internal::make_wrapped_pfpx<Scalar, Device, PhysicalDim, TopologyDim>(compute_hessian, pfpx, make_spsd);
    parallel().vmap(wrapped_pfpx, local_stiffness, deform_grad, pfpx);

    mp::sparse::basic_gather_operator<Scalar, Device, 0> gather_op{
      sysmat_nonzeros,              // hessian on vertices,
      local_stiffness.flatten(),    // hessian on elements
      hessian_gather_info_.desc(),  // cached gather information
      time_step_ * time_step_};     // alpha = dt^2
    parallel().run(gather_op);

    // filter out the dirichlet boundary conditions.
    boundary_type enforce(mesh_.const_view(), dof_type(), dbc_values());
    enforce.hessian(parallel(), sysmatrix());
  }

  bool check_convergence() {
    auto force = forces_.view();
    Scalar norm = blas().norm(force);
    return norm < gradient_convergence_threshold_abs_;
  }

private:
  ////////// Basic //////////
  mesh_type mesh_;                    ///< state of the mesh at zero step.
  host_mesh_type host_mesh_;          ///< host mesh for more efficient computes.
  batched_dof_type_buffer dof_type_;  ///< boundary indicator
  vertex_buffer dbc_values_;          ///< dirichlet boundary condition values.
  vertex_buffer deformation_;         ///< deform at current step.
  vertex_buffer prev_deform_;         ///< deform at previous step.
  vertex_buffer inertia_deform_;      ///< deformation with one step inertia term.
  vertex_buffer next_deform_;         ///< deform at next step, or the intermediate step.
  vertex_buffer velocity_;            ///< velocity at current step.
  vertex_buffer ext_accel_;           ///< external acceleration.
  vertex_buffer forces_;              ///< forces at current step. (Residual of Dynamic System)
  vertex_buffer temp_buffer_;         ///< temporary buffer for intermediate results.
  sys_matrix sysmatrix_;              ///< system matrix at current step.

  ////////// Elasticity //////////
  Scalar youngs_;                           ///< Young's modulus. (if uniform)
  Scalar poisson_;                          ///< Poisson's ratio. (if uniform)
  rest_volume_buffer rest_volume_;          ///< rest volume of each element.
  dminv_buffer dminv_;                      ///< See "Dynamic Deformables", map x->F
  pfpx_buffer pfpx_;                        ///< derivative of DeformGrad wrt x.
  deform_grad_buffer deform_grad_;          ///< deformation gradient.
  local_energy_buffer local_energy_;        ///< element local energy
  local_stress_buffer local_stress_;        ///< element local force
  local_force_buffer local_force_;          ///< element (converted on vertex) local force
  local_stiffness_buffer local_stiffness_;  ///< element local stiffness matrix

  ////////// Mass //////////
  Scalar density_;                   ///< density. (if uniform)
  sys_matrix mass_matrix_;           ///< mass matrix.
  sys_matrix mass_matrix_filtered_;  ///< mass matrix with dirichlet boundary conditions.

  mp::sparse::basic_gather_info<Scalar, Device> stress_gather_info_;   // gathers stress from elements to nodes.
  mp::sparse::basic_gather_info<Scalar, Device> hessian_gather_info_;  // gathers stress from elements to nodes.

  ////////// Time Integration information //////////
  Scalar time_step_;      ///< time step size.
  Scalar world_time_{0};  ///< current world time.

  Scalar gradient_convergence_threshold_{1e-2};      // relative threshold for gradient convergence.
  Scalar gradient_convergence_threshold_abs_{1e-2};  // absolute threshold for gradient convergence.
  Scalar damping_{0.0};                              // damping for velocity.

  SparseBlas sys_bl_;
  SparseBlas mass_bl_;
  SparseBlas mass_filtered_bl_;
  Blas blas_;
  ParImpl pf_;
};

class time_step_solver_forward_euler : public basic_time_step_solver<time_step_solver_forward_euler> {
public:
  template <typename Scalar, typename Device,                               //
            index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
            typename SparseBlas, typename Blas, typename ParImpl>
  void solve_impl(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>& s) {
    s.update_energy_and_gradients(false);
    // u_next = u_iner + 1/2 dt * dt * f
    auto u_next = s.next_deform();  // the time step sets to inertia deform by default.
    auto f = s.forces();
    s.blas().axpy(0.5 * s.time_step() * s.time_step(), f, u_next);
  }

  template <typename Scalar, typename Device,                               //
            index_t PhysicalDim, index_t TopologyDim, typename ElastModel,  //
            typename SparseBlas, typename Blas, typename ParImpl>
  void reset_impl(basic_time_step<Scalar, Device, PhysicalDim, TopologyDim, ElastModel, SparseBlas, Blas, ParImpl>&) {
    // nothing to reset.
  }
};

}  // namespace ssim::fem
