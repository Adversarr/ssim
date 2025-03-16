/**
 * @brief Represents the minimum information about the mesh vertices and its topology.
 *
 * @note Here are some examples of the mesh topology:
 *      PhysicalDim = 2, TopologyDim = 3 => 2D, triangle mesh
 *      PhysicalDim = 3, TopologyDim = 4 => 3D, tetrahedron mesh
 *      PhysicalDim = 3, TopologyDim = 3 => 3D, triangle mesh
 *
 * @tparam Scalar
 * @tparam Device
 * @tparam PhysicalDim
 * @tparam IntrinsicDim
 * @tparam TopologyDim
 */

#pragma once
#include <mathprim/core/buffer.hpp>
#include <mathprim/core/view.hpp>

#include "ssim/defines.hpp"

namespace ssim {
template <typename Scalar, typename Device, index_t PhysicalDim, index_t TopologyDim>
class basic_unstructured_view {
  static_assert(PhysicalDim > 0 && TopologyDim > 0, "PhysicalDim and TopologyDim must be positive integers.");
public:
  using scalar_type = Scalar;
  using device_type = Device;
  static constexpr index_t physical_dim = PhysicalDim;
  static constexpr index_t topology_dim = TopologyDim;
  static constexpr bool is_const = std::is_const_v<Scalar>;
  using maybe_const_index = std::conditional_t<is_const, const index_t, index_t>;

  using vertex_type = mp::contiguous_view<Scalar, mp::shape_t<physical_dim>, Device>;
  using cell_type = mp::contiguous_view<maybe_const_index, mp::shape_t<topology_dim>, Device>;
  using batched_vertex = mp::batched<vertex_type>;
  using batched_cell = mp::batched<cell_type>;

  basic_unstructured_view() = default;
  SSIM_INTERNAL_ENABLE_ALL_CTOR(basic_unstructured_view);
  basic_unstructured_view(const batched_vertex& vertices, const batched_cell& cells) :
      vertices_(vertices), cells_(cells) {}

  SSIM_PRIMFUNC const batched_vertex& vertices() const noexcept { return vertices_; }
  SSIM_PRIMFUNC const batched_cell& cells() const noexcept { return cells_; }

  SSIM_PRIMFUNC index_t num_vertices() const noexcept { return vertices_.shape(0); }
  SSIM_PRIMFUNC index_t num_cells() const noexcept { return cells_.shape(0); }

  SSIM_PRIMFUNC basic_unstructured_view<std::add_const_t<Scalar>, Device, PhysicalDim, TopologyDim> as_const()
      const noexcept {
    return {vertices_.as_const(), cells_.as_const()};
  }

private:
  batched_vertex vertices_;
  batched_cell cells_;
};

template <typename Scalar, typename Device, index_t PhysicalDim, index_t TopologyDim>
class basic_unstructured {
  static_assert(!std::is_const_v<Scalar>, "Scalar type must be non-const.");
public:
  using scalar_type = Scalar;
  using device_type = Device;
  static constexpr index_t physical_dim = PhysicalDim;
  static constexpr index_t topology_dim = TopologyDim;
  using view_type = basic_unstructured_view<Scalar, Device, PhysicalDim, TopologyDim>;
  using const_view_type = basic_unstructured_view<const Scalar, Device, PhysicalDim, TopologyDim>;

  using vertex_type = typename view_type::vertex_type;
  using cell_type = typename view_type::cell_type;
  using batched_vertex = typename view_type::batched_vertex;
  using batched_cell = typename view_type::batched_cell;

  using batched_vertex_buffer = mp::to_buffer_t<batched_vertex>;
  using batched_cell_buffer = mp::to_buffer_t<batched_cell>;

  basic_unstructured() = default;
  SSIM_INTERNAL_COPY(basic_unstructured, delete);
  SSIM_INTERNAL_MOVE(basic_unstructured, default);

  basic_unstructured(index_t num_vertices, index_t num_cells) :
      vertices_(mp::make_buffer<Scalar, Device>(num_vertices, mp::holder<physical_dim>{})),
      cells_(mp::make_buffer<index_t, Device>(num_cells, mp::holder<topology_dim>{})) {}

  view_type view() noexcept { return {vertices_.view(), cells_.view()}; }
  const_view_type view() const noexcept { return const_view(); }
  const_view_type const_view() const noexcept { return {vertices_.const_view(), cells_.const_view()}; }

  index_t num_vertices() const noexcept { return vertices_.shape(0); }
  index_t num_cells() const noexcept { return cells_.shape(0); }

  auto vertices() noexcept { return vertices_.view(); }
  auto cells() noexcept { return cells_.view(); }

private:
  batched_vertex_buffer vertices_;
  batched_cell_buffer cells_;
};

template <typename Scalar, typename Device>
using line_mesh_view = basic_unstructured_view<Scalar, Device, 1, 2>;
template <typename Scalar, typename Device>
using tri_mesh_view = basic_unstructured_view<Scalar, Device, 2, 3>;
template <typename Scalar, typename Device>
using manifold_mesh_view = basic_unstructured_view<Scalar, Device, 3, 3>;
template <typename Scalar, typename Device>
using tetmesh_view = basic_unstructured_view<Scalar, Device, 3, 4>;

template <typename Scalar, typename Device>
using line_mesh = basic_unstructured<Scalar, Device, 1, 2>;
template <typename Scalar, typename Device>
using tri_mesh = basic_unstructured<Scalar, Device, 2, 3>;
template <typename Scalar, typename Device>
using manifold_mesh = basic_unstructured<Scalar, Device, 3, 3>;
template <typename Scalar, typename Device>
using tet_mesh = basic_unstructured<Scalar, Device, 3, 4>;
}  // namespace ssim
