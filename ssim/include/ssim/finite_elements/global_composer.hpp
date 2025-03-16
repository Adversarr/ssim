#pragma once
#include <map>
#include <mathprim/sparse/gather.hpp>
#include "ssim/defines.hpp"
#include "ssim/mesh/basic_mesh.hpp"

namespace ssim::fem {

// Logically, it converts bcoo to csr, by defining a gather operation.
template <typename Scalar, typename Device, index_t PhysicalDim, index_t TopologyDim>
class local_global_composer {
public:
  using scalar_type = Scalar;
  using device_type = Device;
  static constexpr index_t physical_dim = PhysicalDim;
  static constexpr index_t topology_dim = TopologyDim;
  static constexpr index_t dofs_per_node = PhysicalDim;
  static constexpr index_t hessian_nrows = topology_dim * dofs_per_node;

  using local_stiffness = mp::contiguous_view<Scalar, mp::shape_t<hessian_nrows, hessian_nrows>, Device>;
  using const_local_matrix = mp::contiguous_view<const Scalar, mp::shape_t<hessian_nrows, hessian_nrows>, Device>;

  using mesh_view = basic_unstructured_view<Scalar, Device, PhysicalDim, TopologyDim>;
  using vertex_type = typename mesh_view::vertex_type;
  using cell_type = typename mesh_view::cell_type;
  using batched_vertex = typename mesh_view::batched_vertex;
  using batched_cell = typename mesh_view::batched_cell;
  
  using const_mesh_view = basic_unstructured_view<const Scalar, Device, PhysicalDim, TopologyDim>;
  using const_vertex = typename const_mesh_view::vertex_type;
  using const_cell = typename const_mesh_view::cell_type;
  using const_batched_vertex = typename const_mesh_view::batched_vertex;
  using const_batched_cell = typename const_mesh_view::batched_cell;
  using cell_value = mp::contiguous_vector_view<Scalar, Device>;
  using const_cell_value = mp::contiguous_vector_view<const Scalar, Device>;

  mp::sparse::basic_gather_info<Scalar, Device> force(const_mesh_view mesh, const_cell_value cell_values) {
    auto cell = mesh.cells();
    auto cell_on_cpu_buf = mp::make_buffer<index_t>(cell.shape());
    auto cell_values_on_cpu_buf = mp::make_buffer<Scalar>(cell_values.shape());
    index_t num_cells = cell.shape(0);
    auto cell_on_cpu = cell_on_cpu_buf.view();
    auto cell_values_on_cpu = cell_values_on_cpu_buf.view();
    mp::copy(cell_on_cpu, cell);
    mp::copy(cell_values_on_cpu, cell_values);

    std::vector<mp::sparse::entry<Scalar>> gather_local_stiffness;
    // Gather Operation support vectorization over dof_per_node.
    auto local_stress_shape = mp::make_shape(num_cells, topology_dim);
    auto local_stress_stride = mp::make_default_stride(local_stress_shape);

    for (index_t elem_id = 0; elem_id < num_cells; ++elem_id) {
      auto cell = cell_on_cpu[elem_id];
      Scalar cell_value = cell_values_on_cpu[elem_id];
      for (auto [i] : mp::make_shape(topology_dim)) {
        const index_t dst = cell[i];
        const index_t src = mp::sub2ind(local_stress_stride, mp::index_array<2>{elem_id, i});
        gather_local_stiffness.emplace_back(dst, src, cell_value);
      }
    }

    const index_t num_outputs = mesh.num_vertices();
    const index_t num_inputs = local_stress_shape.numel();
    mp::sparse::basic_gather_info<Scalar, Device> result(num_outputs, num_inputs, true);
    result.set_from_triplets(gather_local_stiffness.begin(), gather_local_stiffness.end());
    return result;
  }

  mp::sparse::basic_gather_info<Scalar, Device> hessian(const_mesh_view mesh, const_cell_value cell_values) {
    auto vert = mesh.vertices();
    auto cell = mesh.cells();
    index_t total_dofs = vert.size();
    index_t num_cells = mesh.num_cells();

    auto cell_on_cpu_buf = mp::make_buffer<index_t>(cell.shape());
    auto cell_values_on_cpu_buf = mp::make_buffer<Scalar>(cell_values.shape());
    auto cell_on_cpu = cell_on_cpu_buf.view();
    auto cell_values_on_cpu = cell_values_on_cpu_buf.view();
    mp::copy(cell_on_cpu, cell);
    mp::copy(cell_values_on_cpu, cell_values);
    // 1. compute the output csr matrix topology
    std::vector<mp::sparse::entry<Scalar>> topo;
    for (auto cell: cell_on_cpu) {
      for (auto [i, j] : mp::make_shape(topology_dim, topology_dim)) {
        for (auto [i_dof, j_dof] : mp::make_shape(dofs_per_node, dofs_per_node)) {
          const index_t row = cell[i] * dofs_per_node + i_dof;
          const index_t col = cell[j] * dofs_per_node + j_dof;
          topo.emplace_back(row, col, 0);
        }
      }
    }
    auto hessian_topo = mp::sparse::make_from_coos<Scalar, mathprim::sparse::sparse_format::csr>(
        mp::sparse::make_from_triplets<Scalar>(topo.begin(), topo.end(), total_dofs, total_dofs));

    std::map<std::pair<index_t, index_t>, index_t> ij_to_offset;
    auto task = hessian_topo.view().visit(
      [&, cnt = static_cast<index_t>(0)](index_t row, index_t col, Scalar /* value */) mutable {
        ij_to_offset[std::make_pair(row, col)] = cnt++;
      });
    mp::par::seq().run(task);

    // 2. for each local stiffness matrix's element, computes where should we put into.
    std::vector<mp::sparse::entry<Scalar>> gather_local_stiffness;
    auto local_stiff_shape = mp::make_shape(num_cells, hessian_nrows, hessian_nrows);
    auto local_stiff_stride = mp::make_default_stride(local_stiff_shape);
    for (index_t elem_id = 0; elem_id < cell_on_cpu.shape(0); ++elem_id) {
      auto cell = cell_on_cpu[elem_id];
      Scalar cell_value = cell_values_on_cpu[elem_id];
      for (auto [i, j] : mp::make_shape(topology_dim, topology_dim)) {
        for (auto [i_dof, j_dof] : mp::make_shape(dofs_per_node, dofs_per_node)) {
          const index_t row = cell[i] * dofs_per_node + i_dof;         // ... of global matrix
          const index_t col = cell[j] * dofs_per_node + j_dof;         // ... of global matrix
          const index_t dst = ij_to_offset.at(std::make_pair(row, col));  // ... of global csr matrix
          mp::index_array<3> idx{elem_id, i * dofs_per_node + i_dof, j * dofs_per_node + j_dof};
          const index_t src = mp::sub2ind(local_stiff_stride, idx);
          gather_local_stiffness.emplace_back(dst, src, cell_value);
        }
      }
    }

    // 3. convert to gather info.
    const index_t num_outputs = hessian_topo.nnz();
    const index_t num_inputs = local_stiff_shape.numel();
    mp::sparse::basic_gather_info<Scalar, Device> result(num_outputs, num_inputs, true);
    result.set_from_triplets(gather_local_stiffness.begin(), gather_local_stiffness.end());

    return result;
  }
};


}