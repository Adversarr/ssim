#pragma once
#include <mathprim/parallel/parallel.hpp>
#include <mathprim/supports/eigen_dense.hpp>

#include "ssim/elast/basic_elast.hpp"
#include "ssim/elast/internal_physic.hpp"
#include "ssim/mesh/basic_mesh.hpp"

namespace ssim::fem {


template <typename Scalar, typename Device, index_t PhysicalDim, index_t TopologyDim>
class deformation_gradient {
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

  /// D_m .inv ///
  using dminv_item = mp::contiguous_view<Scalar, mp::shape_t<PhysicalDim, PhysicalDim>, Device>;
  using const_dminv_item = mp::contiguous_view<const Scalar, mp::shape_t<PhysicalDim, PhysicalDim>, Device>;
  using dminv_type = mp::batched<dminv_item>;
  using const_dminv = mp::batched<const_dminv_item>;

  /// dF_dX ///
  using pfpx_item = mp::contiguous_view<Scalar, mp::shape_t<PhysicalDim * PhysicalDim, hessian_nrows>, Device>;
  using const_pfpx_item
      = mp::contiguous_view<const Scalar, mp::shape_t<PhysicalDim * PhysicalDim, hessian_nrows>, Device>;
  using pfpx_type = mp::batched<pfpx_item>;
  using const_pfpx = mp::batched<const_pfpx_item>;
  
  /// F ///
  using def_grad_item = elast::def_grad_t<Scalar, Device, PhysicalDim>;
  using const_def_grad_item = elast::def_grad_t<const Scalar, Device, PhysicalDim>;
  using def_grad_type = mp::batched<def_grad_item>;

  /// |dm| ///
  using rest_volume = mp::contiguous_vector_view<Scalar, Device>;
  using const_rest_volume = mp::contiguous_vector_view<const Scalar, Device>;

  struct dminv_task : public mp::par::basic_task<dminv_task> {
    dminv_task(dminv_type dst, const_mesh_view mesh) : dst(dst), mesh(mesh) {}
    dminv_type dst;
    const_mesh_view mesh;

    SSIM_PRIMFUNC void operator()(index_t elem_id) const noexcept {
      auto cell = mesh.cells()[elem_id];
      auto dst_elem = dst[elem_id];
      operator()(dst_elem, cell);
    }

    SSIM_PRIMFUNC void operator()(dminv_item dst_elem, const_cell cell) const noexcept {
      using mat = Eigen::Matrix<Scalar, PhysicalDim, PhysicalDim>;
      using mp::eigen_support::cmap;
      using mp::eigen_support::view;
      auto vert = mesh.vertices();
      index_t i = cell[0], j = cell[1], k = cell[2];
      auto x0 = cmap(vert[i]);
      auto x1 = cmap(vert[j]);
      auto x2 = cmap(vert[k]);
      mat dm;
      dm.col(0) = x1 - x0;
      dm.col(1) = x2 - x0;
      if constexpr (PhysicalDim == 3) {
        index_t l = cell[3];
        auto x3 = cmap(vert[l]);
        dm.col(2) = x3 - x0;
      }
      cmap(dst_elem).transpose() = dm.inverse().eval();
    }

    template <typename ParImpl>
    void run_impl(const mp::par::parfor<ParImpl>& impl) const noexcept {
      impl.run(mesh.num_cells(), *this);
    }
  };

  struct def_grad_task : public mp::par::basic_task<def_grad_task> {
    def_grad_task(def_grad_type dst, const_mesh_view mesh, const_dminv dminv, const_batched_vertex deform) :
        dst(dst), mesh(mesh), dminv(dminv), deform_(deform) {}
    def_grad_type dst;
    const_mesh_view mesh;
    const_dminv dminv;
    const_batched_vertex deform_;

    SSIM_PRIMFUNC void operator()(index_t elem_id) const noexcept {
      auto cell = mesh.cells()[elem_id];
      auto dst_elem = dst[elem_id];
      auto dminv_elem = dminv[elem_id];
      operator()(dst_elem, cell, dminv_elem);
    }

    SSIM_PRIMFUNC void operator()(def_grad_item dst_elem, const_cell cell, const_dminv_item dminv_elem) const noexcept {
      using mat = Eigen::Matrix<Scalar, PhysicalDim, PhysicalDim>;
      using mp::eigen_support::cmap;
      using mp::eigen_support::view;
      auto vert = mesh.vertices();
      index_t i = cell[0], j = cell[1], k = cell[2];
      auto x0 = cmap(vert[i]) + cmap(deform_[i]);
      auto x1 = cmap(vert[j]) + cmap(deform_[j]);
      auto x2 = cmap(vert[k]) + cmap(deform_[k]);
      mat dm;
      dm.col(0) = x1 - x0;
      dm.col(1) = x2 - x0;
      if constexpr (PhysicalDim == 3) {
        index_t l = cell[3];
        auto x3 = cmap(vert[l]) + cmap(deform_[l]);
        dm.col(2) = x3 - x0;
      }
      cmap(dst_elem).transpose() = dm * cmap(dminv_elem).transpose();
    }

    template <typename ParImpl>
    void run_impl(const mp::par::parfor<ParImpl>& impl) const noexcept {
      impl.run(mesh.num_cells(), *this);
    }
  };

  struct pfpx_task : public mp::par::basic_task<pfpx_task> {
    pfpx_task(pfpx_type dst, const_mesh_view mesh, const_dminv dminv) : dst(dst), mesh(mesh), dminv(dminv) {}
    pfpx_type dst;
    const_mesh_view mesh;
    const_dminv dminv;

    SSIM_PRIMFUNC void operator()(index_t elem_id) const noexcept {
      auto dst_elem = dst[elem_id];
      auto dminv_elem = dminv[elem_id];
      operator()(dst_elem, dminv_elem);
    }

    SSIM_PRIMFUNC void operator()(pfpx_item pfpx, const_dminv_item dminv_elem) const noexcept {
      using mp::eigen_support::cmap;
      using mp::eigen_support::view;
      for (auto [i, j] : pfpx.shape()) {
        pfpx(i, j) = 0;
      }
      if constexpr (PhysicalDim == 2) {
        elast::internal::compute_pfpx_2<Scalar, pfpx_item>(cmap(dminv_elem).transpose(), pfpx);
      } else {
        elast::internal::compute_pfpx_3<Scalar, pfpx_item>(cmap(dminv_elem).transpose(), pfpx);
      }
    }

    template <typename ParImpl>
    void run_impl(const mp::par::parfor<ParImpl>& impl) const noexcept {
      impl.vmap(*this, dst, dminv);
    }
  };

  /// @brief create a task to compute inv(Dm) for each element.
  dminv_task compute_dminv(dminv_type dst) const noexcept { return {dst, mesh_}; }

  /// @brief create a task to compute F for each element.
  pfpx_task compute_pfpx(pfpx_type dst, const_dminv dminv) const noexcept { return {dst, mesh_, dminv}; }

  def_grad_task compute_def_grad(def_grad_type dst, const_dminv dminv, const_batched_vertex deform) const noexcept {
    return {dst, mesh_, dminv, deform};
  }

  explicit deformation_gradient(const_mesh_view mesh) : mesh_(std::move(mesh)) {}
  SSIM_INTERNAL_ENABLE_ALL_CTOR(deformation_gradient);

  const_mesh_view mesh_;
};

}  // namespace ssim::fem
