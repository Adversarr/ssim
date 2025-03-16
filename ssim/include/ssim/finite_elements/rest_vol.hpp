#pragma once
#include <mathprim/parallel/parallel.hpp>
#include <mathprim/supports/eigen_dense.hpp>

#include "ssim/mesh/basic_mesh.hpp"

namespace ssim::fem {

template <typename Scalar, typename Device, index_t PhysicalDim, index_t TopologyDim>
class rest_vol_task : public mp::par::basic_task<rest_vol_task<Scalar, Device, PhysicalDim, TopologyDim>> {
public:
  static_assert(PhysicalDim == 2 || PhysicalDim == 3, "PhysicalDim must be 2 or 3.");
  using scalar_type = Scalar;
  using device_type = Device;
  static constexpr index_t physical_dim = PhysicalDim;
  static constexpr index_t topology_dim = TopologyDim;
  static constexpr index_t dofs_per_node = PhysicalDim;
  static constexpr index_t hessian_nrows = topology_dim * dofs_per_node;
  using const_mesh_view = basic_unstructured_view<const Scalar, Device, PhysicalDim, TopologyDim>;
  using mesh_view = basic_unstructured_view<Scalar, Device, PhysicalDim, TopologyDim>;
  using vertex_type = typename mesh_view::vertex_type;
  using const_vertex = typename const_mesh_view::vertex_type;
  using cell_type = typename mesh_view::cell_type;
  using const_cell = typename const_mesh_view::cell_type;

  using batched_vertex = typename mesh_view::batched_vertex;
  using batched_cell = typename mesh_view::batched_cell;
  using const_batched_vertex = typename const_mesh_view::batched_vertex;
  using const_batched_cell = typename const_mesh_view::batched_cell;
  /// |dm| ///
  using rest_volume = mp::contiguous_vector_view<Scalar, Device>;
  using const_rest_volume = mp::contiguous_vector_view<const Scalar, Device>;

  SSIM_PRIMFUNC void operator() (Scalar& out, const_cell cell) const noexcept {
    index_t i = cell[0], j = cell[1], k = cell[2];
    using mat = Eigen::Matrix<Scalar, PhysicalDim, PhysicalDim>;
    auto vert = mesh_.vertices();
    using mp::eigen_support::cmap;
    using mp::eigen_support::view;
    auto x0 = cmap(vert[i]);
    auto x1 = cmap(vert[j]);
    auto x2 = cmap(vert[k]);
    mat dm;
    dm.col(0) = x1 - x0;
    dm.col(1) = x2 - x0;
    if constexpr (PhysicalDim == 2) {
      out = dm.determinant() / 2;
    } else {
      index_t l = cell[3];
      auto x3 = cmap(vert[l]);
      dm.col(2) = x3 - x0;
      out = dm.determinant() / 6;
    }
  }

  template <typename ParImpl>
  void run_impl(const mp::par::parfor<ParImpl>& pf) const noexcept {
    pf.vmap(*this, rest_vol_, mesh_.cells());
  }

  explicit rest_vol_task(const_mesh_view mesh, rest_volume rest_vol)
      : mesh_(mesh), rest_vol_(rest_vol) {}
  SSIM_INTERNAL_ENABLE_ALL_CTOR(rest_vol_task);

  const_mesh_view mesh_;
  rest_volume rest_vol_;
};

}  // namespace ssim::fem