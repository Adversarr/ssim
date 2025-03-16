#pragma once

#include "ssim/mesh/basic_mesh.hpp"
#include "boundary.hpp"

namespace ssim::fem {

template <typename Scalar, typename Device, index_t PhysicalDim, index_t TopologyDim, typename ElastModel>
class basic_time_step;

class basic_time_step_solver; // TODO:

template <typename Scalar, typename Device,          //
          index_t PhysicalDim, index_t TopologyDim,  //
          typename ElastModel>
class basic_time_step {
public:
  using scalar_type = Scalar;
  using device_type = Device;
  static constexpr index_t physical_dim = PhysicalDim;
  static constexpr index_t topology_dim = TopologyDim;
  static constexpr index_t dofs_per_node = PhysicalDim;
  static constexpr index_t hessian_nrows = topology_dim * dofs_per_node;

  // Each term, can have its own local stiffness matrix, force vector, energy value.
  using local_stiffness = mp::contiguous_view<Scalar, mp::shape_t<hessian_nrows, hessian_nrows>, Device>;
  using const_local_matrix = mp::contiguous_view<const Scalar, mp::shape_t<hessian_nrows, hessian_nrows>, Device>;
  using local_force = mp::contiguous_view<Scalar, mp::shape_t<topology_dim, dofs_per_node>, Device>;
  using const_local_force = mp::contiguous_view<const Scalar, mp::shape_t<topology_dim, dofs_per_node>, Device>;

  using batched_local_stiffness = mp::batched<local_stiffness>;
  using batched_local_force = mp::batched<local_force>;
  using batched_local_energy = mp::contiguous_vector_view<Scalar, Device>;

  using local_stiffness_buffer = mp::to_buffer_t<local_stiffness>;
  using local_force_buffer = mp::to_buffer_t<local_force>;
  using local_energy_buffer = mp::to_buffer_t<batched_local_energy>;

  using const_batched_local_stiffness = mp::batched<const_local_matrix>;
  using const_batched_local_force = mp::batched<const_local_force>;
  using const_batched_local_energy = mp::contiguous_vector_view<const Scalar, Device>;

  // For mass matrix, we can have a single mass matrix for all dofs.
  using node_mass_type = mp::contiguous_vector_view<Scalar, Device>;
  using const_node_mass = mp::contiguous_vector_view<const Scalar, Device>;
  using node_mass_buffer = mp::to_buffer_t<node_mass_type>;

  using sys_matrix = sparse_matrix<Scalar, Device>;
  using sys_matrix_view = sparse_view<Scalar, Device>;

  // Mesh
  using mesh_type = basic_unstructured<Scalar, Device, PhysicalDim, TopologyDim>;
  using const_mesh_view = basic_unstructured_view<const Scalar, Device, PhysicalDim, TopologyDim>;
  using mesh_view = basic_unstructured_view<Scalar, Device, PhysicalDim, TopologyDim>;
  using boundary_type = boundary_condition<Scalar, Device, PhysicalDim, TopologyDim, dofs_per_node>;
  using vertex_type = typename mesh_type::vertex_type;
  using cell_type = typename mesh_type::cell_type;
  using batched_vertex = typename mesh_type::batched_vertex;
  using batched_cell = typename mesh_type::batched_cell;
  using vertex_buffer = typename mesh_type::batched_vertex_buffer;
  using cell_buffer = typename mesh_type::batched_cell_buffer;

  // Elast
  using rest_volume = mp::contiguous_vector_view<Scalar, Device>;
  using const_rest_volume = mp::contiguous_vector_view<const Scalar, Device>;
  using rest_volume_buffer = mp::to_buffer_t<rest_volume>;
  using dminv_item = mp::contiguous_view<Scalar, mp::shape_t<PhysicalDim, PhysicalDim>, Device>;
  using dminv_type = mp::batched<dminv_item>;
  using dminv_buffer = mp::to_buffer_t<dminv_type>;
  using pfpx_item = mp::contiguous_view<Scalar, mp::shape_t<PhysicalDim * PhysicalDim, hessian_nrows>, Device>;
  using pfpx_type = mp::batched<pfpx_item>;
  using pfpx_buffer = mp::to_buffer_t<pfpx_type>;

  explicit basic_time_step(mesh_type mesh) : mesh_(std::move(mesh)) {}
  SSIM_INTERNAL_ENABLE_ALL_CTOR(basic_time_step);

  SSIM_PRIMFUNC const mesh_type& mesh() const noexcept { return mesh_; }

private:
  ////////// Basic //////////
  Scalar time_step_;           ///< time step size.
  mesh_type mesh_;             ///< state of the mesh at zero step.
  vertex_buffer deformation_;  ///< deform at current step.
  vertex_buffer prev_deform_;  ///< deform at previous step.
  vertex_buffer next_deform_;  ///< deform at next step, or the intermediate step.
  vertex_buffer velocity_;     ///< velocity at current step.
  vertex_buffer ext_accel_;    ///< external acceleration.
  vertex_buffer forces_;       ///< forces at current step. (Residual of Dynamic System)
  boundary_type boundary_;     ///< boundary conditions handler.
  
  sys_matrix system_matrix_;   ///< system matrix at current step.
  ////////// Mass //////////
  node_mass_buffer density_;   ///< density.
  sys_matrix mass_matrix_;     ///< mass matrix.

  ////////// Elasticity //////////
  Scalar youngs_;                           ///< Young's modulus.
  Scalar poisson_;                          ///< Poisson's ratio.
  rest_volume_buffer rest_volume_;          ///< rest volume of each element.
  dminv_buffer dminv_;                      ///< See "Dynamic Deformables", map x->F
  pfpx_buffer pfpx_;                        ///< derivative of DeformGrad wrt x.
  local_energy_buffer local_energy_;        ///< element local energy
  local_force_buffer local_force_;          ///< element local force
  local_stiffness_buffer local_stiffness_;  ///< element local stiffness matrix
};

}  // namespace ssim::fem