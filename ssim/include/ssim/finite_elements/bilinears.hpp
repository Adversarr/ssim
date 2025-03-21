#pragma once
/**
 * @note mathematica code:
 * f3[x_, y_] = y
 * f2[x_, y_] = x
 * f1[x_, y_] = 1 - x - y
 * Basis = {f1, f2, f3}
 * (* For U V *)
 * Table[Integrate[ Integrate[Basis[[i]][x, y] Basis[[j]][x, y], {x, 0, 1 - y}], {y, 0, 1}],{i, 1,
 * 3}, {j, 1, 3}]
 * (* For Du v *)
 * Table[Integrate[Integrate[D[Basis[[i]][x, y], P] Basis[[j]][x, y], {x, 0, 1 - y}], {y, 0, 1}],
 * {i, 1, 3}, {j, 1, 3}, {P, {x, y}}]
 * (* For Du Dv *)
 * Table[Integrate[Integrate[D[Basis[[i]][x, y], P] D[Basis[[j]][x, y], Q], {x, 0, 1 - y}], {y, 0,
 * 1}], {i, 1, 3}, {j, 1, 3}, {P, {x, y}}, {Q, {x, y}}]
 *
 */
#include <mathprim/supports/eigen_dense.hpp>

#include "ssim/defines.hpp"
#include "ssim/finite_elements/def_grad.hpp"
#include "ssim/mesh/basic_mesh.hpp"

namespace ssim::fem {

namespace internal {
constexpr double p12_element_f_f[3][3]
    = {{1.0 / 12, 1.0 / 24, 1.0 / 24}, {1.0 / 24, 1.0 / 12, 1.0 / 24}, {1.0 / 24, 1.0 / 24, 1.0 / 12}};

// [K][i][j] = Integrate [(diff phi_i) / (diff x_K), phi_j]
constexpr double p12_element_pfpx_f[3][3][2]
    = {{{-(1.0 / 6), -(1.0 / 6)}, {-(1.0 / 6), -(1.0 / 6)}, {-(1.0 / 6), -(1.0 / 6)}},
       {{1.0 / 6, 0}, {1.0 / 6, 0}, {1.0 / 6, 0}},
       {{0, 1.0 / 6}, {0, 1.0 / 6}, {0, 1.0 / 6}}};

// [i][j][k][l] = Integrate [(diff phi_i) / (diff x_k), (diff phi_j) / (diff x_l)]
constexpr double p12_pfpx_pfpx[3][3][2][2] = {
  {{{1.0 / 2, 1.0 / 2}, {1.0 / 2, 1.0 / 2}}, {{-(1.0 / 2), 0}, {-(1.0 / 2), 0}}, {{0, -(1.0 / 2)}, {0, -(1.0 / 2)}}},
  {{{-(1.0 / 2), -(1.0 / 2)}, {0, 0}}, {{1.0 / 2, 0}, {0, 0}}, {{0, 1.0 / 2}, {0, 0}}},
  {{{0, 0}, {-(1.0 / 2), -(1.0 / 2)}}, {{0, 0}, {1.0 / 2, 0}}, {{0, 0}, {0, 1.0 / 2}}}};



template <typename Scalar>
struct p1_element_2d {
  SSIM_PRIMFUNC p1_element_2d() {
    jacobi_.setIdentity();
    inverse_jacobi_.setIdentity();
    det_jacobi_ = 1;
  }

  explicit SSIM_PRIMFUNC p1_element_2d(Eigen::Matrix<Scalar, 3, 3> const& nodes) {
    // 1. find the plane.
    Eigen::Vector3<Scalar> v10 = nodes.col(1) - nodes.col(0);
    Eigen::Vector3<Scalar> v20 = nodes.col(2) - nodes.col(0);
    jacobi_(0, 0) = v10.norm();
    jacobi_(0, 1) = 0;
    jacobi_(1, 0) = v10.dot(v20) / jacobi_(0, 0);
    jacobi_(1, 1) = sqrt(v20.squaredNorm() - jacobi_(1, 0) * jacobi_(1, 0));
    inverse_jacobi_ = jacobi_.inverse();
    det_jacobi_ = abs(jacobi_.determinant());
  }

  explicit SSIM_PRIMFUNC p1_element_2d(Eigen::Matrix<Scalar, 2, 3> const& nodes) {
    // jacobi_ << (nodes[1] - nodes[0]), (nodes[2] - nodes[0]);
    jacobi_.col(0) = nodes.col(1) - nodes.col(0);
    jacobi_.col(1) = nodes.col(2) - nodes.col(0);
    inverse_jacobi_ = jacobi_.inverse();
    det_jacobi_ = abs(jacobi_.determinant());
  }

  SSIM_PRIMFUNC Scalar integrate_f_f(index_t i, index_t j) const noexcept {
    return det_jacobi_ * p12_element_f_f[i][j];
  }

  SSIM_PRIMFUNC Scalar integrate_pf_f(index_t i, index_t j, index_t i_pf) const noexcept {
    Scalar pfi_pu = p12_element_pfpx_f[0][i_pf][j];
    Scalar pfi_pv = p12_element_pfpx_f[1][i_pf][j];
    return det_jacobi_ * (pfi_pu * inverse_jacobi_(0, i) + pfi_pv * inverse_jacobi_(1, i));
  }

  SSIM_PRIMFUNC Scalar integrate_pf_pf(index_t i, index_t j, index_t k, index_t l) const noexcept {
    SSIM_ASSERT(i < 3 && j < 3 && k < 2 && l < 2);
    Eigen::Matrix2<Scalar> pfi_pu_pu;
    pfi_pu_pu << p12_pfpx_pfpx[i][j][0][0], p12_pfpx_pfpx[i][j][0][1], p12_pfpx_pfpx[i][j][1][0],
        p12_pfpx_pfpx[i][j][1][1];
    auto puv_pxk = inverse_jacobi_.col(k);
    auto puv_pxl = inverse_jacobi_.col(l);
    return det_jacobi_ * puv_pxk.dot(pfi_pu_pu * puv_pxl);
  }

  Eigen::Matrix2<Scalar> jacobi_;
  Eigen::Matrix2<Scalar> inverse_jacobi_;
  Scalar det_jacobi_;
};

template <typename Scalar>
struct p1_element_3d {
  static constexpr double p13_pfpx_f[4][4][3] = {{{-(1.0 / 24), -(1.0 / 24), -(1.0 / 24)},
                                           {-(1.0 / 24), -(1.0 / 24), -(1.0 / 24)},
                                           {-(1.0 / 24), -(1.0 / 24), -(1.0 / 24)},
                                           {-(1.0 / 24), -(1.0 / 24), -(1.0 / 24)}},
                                          {{1.0 / 24, 0, 0}, {1.0 / 24, 0, 0}, {1.0 / 24, 0, 0}, {1.0 / 24, 0, 0}},
                                          {{0, 1.0 / 24, 0}, {0, 1.0 / 24, 0}, {0, 1.0 / 24, 0}, {0, 1.0 / 24, 0}},
                                          {{0, 0, 1.0 / 24}, {0, 0, 1.0 / 24}, {0, 0, 1.0 / 24}, {0, 0, 1.0 / 24}}};

  static constexpr double p13_pfpx_pfpx[4][4][3][3]
      = {{{{1.0 / 6, 1.0 / 6, 1.0 / 6}, {1.0 / 6, 1.0 / 6, 1.0 / 6}, {1.0 / 6, 1.0 / 6, 1.0 / 6}},
          {{-(1.0 / 6), 0, 0}, {-(1.0 / 6), 0, 0}, {-(1.0 / 6), 0, 0}},
          {{0, -(1.0 / 6), 0}, {0, -(1.0 / 6), 0}, {0, -(1.0 / 6), 0}},
          {{0, 0, -(1.0 / 6)}, {0, 0, -(1.0 / 6)}, {0, 0, -(1.0 / 6)}}},
         {{{-(1.0 / 6), -(1.0 / 6), -(1.0 / 6)}, {0, 0, 0}, {0, 0, 0}},
          {{1.0 / 6, 0, 0}, {0, 0, 0}, {0, 0, 0}},
          {{0, 1.0 / 6, 0}, {0, 0, 0}, {0, 0, 0}},
          {{0, 0, 1.0 / 6}, {0, 0, 0}, {0, 0, 0}}},
         {{{0, 0, 0}, {-(1.0 / 6), -(1.0 / 6), -(1.0 / 6)}, {0, 0, 0}},
          {{0, 0, 0}, {1.0 / 6, 0, 0}, {0, 0, 0}},
          {{0, 0, 0}, {0, 1.0 / 6, 0}, {0, 0, 0}},
          {{0, 0, 0}, {0, 0, 1.0 / 6}, {0, 0, 0}}},
         {{{0, 0, 0}, {0, 0, 0}, {-(1.0 / 6), -(1.0 / 6), -(1.0 / 6)}},
          {{0, 0, 0}, {0, 0, 0}, {1.0 / 6, 0, 0}},
          {{0, 0, 0}, {0, 0, 0}, {0, 1.0 / 6, 0}},
          {{0, 0, 0}, {0, 0, 0}, {0, 0, 1.0 / 6}}}};

  SSIM_PRIMFUNC p1_element_3d() {
    jacobi_.setIdentity();
    inverse_jacobi_.setIdentity();
    det_jacobi_ = 1;
  }

  explicit SSIM_PRIMFUNC p1_element_3d(Eigen::Matrix<Scalar, 3, 4> const& nodes) {
    jacobi_.col(0) = nodes.col(1) - nodes.col(0);
    jacobi_.col(1) = nodes.col(2) - nodes.col(0);
    jacobi_.col(2) = nodes.col(3) - nodes.col(0);
    inverse_jacobi_ = jacobi_.inverse().eval();
    det_jacobi_ = abs(jacobi_.determinant());
  }

  SSIM_PRIMFUNC Scalar integrate_f_f(index_t i, index_t j) const noexcept {
    static constexpr double p13_f_f[4][4] = {{1.0 / 60, 1.0 / 120, 1.0 / 120, 1.0 / 120},
                                             {1.0 / 120, 1.0 / 60, 1.0 / 120, 1.0 / 120},
                                             {1.0 / 120, 1.0 / 120, 1.0 / 60, 1.0 / 120},
                                             {1.0 / 120, 1.0 / 120, 1.0 / 120, 1.0 / 60}};
    assert(0 <= i && i < 4);
    assert(0 <= j && j < 4);
    return det_jacobi_ * p13_f_f[i][j];
  }

  SSIM_PRIMFUNC Scalar integrate_pf_f(index_t i, index_t j, index_t i_pf) const noexcept {
    Scalar pfi_pu = p13_pfpx_f[0][i_pf][j];
    Scalar pfi_pv = p13_pfpx_f[1][i_pf][j];
    Scalar pfi_pw = p13_pfpx_f[2][i_pf][j];
    return det_jacobi_
           * (pfi_pu * inverse_jacobi_(0, i) + pfi_pv * inverse_jacobi_(1, i) + pfi_pw * inverse_jacobi_(2, i));
  }

  SSIM_PRIMFUNC Scalar integrate_pf_pf(index_t i, index_t j, index_t k, index_t l) const noexcept {
    SSIM_ASSERT(i < 4 && j < 4 && k < 3 && l < 3);
    Eigen::Matrix3<Scalar> pfi_pu_pu;
    pfi_pu_pu << p13_pfpx_pfpx[i][j][0][0], p13_pfpx_pfpx[i][j][0][1], p13_pfpx_pfpx[i][j][0][2],
        p13_pfpx_pfpx[i][j][1][0], p13_pfpx_pfpx[i][j][1][1], p13_pfpx_pfpx[i][j][1][2], p13_pfpx_pfpx[i][j][2][0],
        p13_pfpx_pfpx[i][j][2][1], p13_pfpx_pfpx[i][j][2][2];
    auto puv_pxk = inverse_jacobi_.col(k);
    auto puv_pxl = inverse_jacobi_.col(l);
    return det_jacobi_ * puv_pxk.dot(pfi_pu_pu * puv_pxl);
  }

  Eigen::Matrix3<Scalar> jacobi_;
  Eigen::Matrix3<Scalar> inverse_jacobi_;
  Scalar det_jacobi_;
};

template <typename Scalar, index_t TopologyDim>
struct element_selector {
  using type = void;
};

template <typename Scalar>
struct element_selector<Scalar, 4> {
  using type = p1_element_3d<Scalar>;
};

template <typename Scalar>
struct element_selector<Scalar, 3> {
  using type = p1_element_2d<Scalar>;
};

}  // namespace internal

template <typename Scalar, typename Device, index_t PhysicalDim, index_t TopologyDim, index_t Partial>
struct linear_element_integrator
    : public mp::par::basic_task<linear_element_integrator<Scalar, Device, PhysicalDim, TopologyDim, Partial>> {
  using element_value = mp::contiguous_vector_view<Scalar, Device>;
  using const_element_value = mp::contiguous_vector_view<const Scalar, Device>;
  using const_mesh_view = basic_unstructured_view<const Scalar, Device, PhysicalDim, TopologyDim>;
  using cell_type = typename const_mesh_view::cell_type;
  using vertex_type = typename const_mesh_view::vertex_type;
  using local_mat = Eigen::Matrix<Scalar, TopologyDim, TopologyDim>;
  using element = typename internal::element_selector<Scalar, TopologyDim>::type;

  using dst_view = mp::contiguous_view<Scalar, mp::shape_t<TopologyDim, TopologyDim>, Device>;

  const_mesh_view mesh_;
  Scalar alpha_{1.0};

  explicit linear_element_integrator(const_mesh_view mesh, Scalar alpha) : mesh_(mesh), alpha_(alpha) {}

  template <typename ParImpl>
  void run_impl(const mp::par::parfor<ParImpl>& pf,  //
                mp::batched<dst_view> dst) const noexcept {
    pf.vmap(*this, mesh_.cells(), dst);
  }

  template <typename ParImpl>
  void run_impl(const mp::par::parfor<ParImpl>& pf,  //
                mp::batched<dst_view> dst,
                const_element_value weights) const noexcept {
    pf.vmap(*this, mesh_.cells(), dst, weights);
  }

  SSIM_PRIMFUNC void operator()(cell_type cell, dst_view dst, Scalar weight = 1) const noexcept {
    using mp::eigen_support::cmap;
    auto verts = mesh_.vertices();
    Eigen::Matrix<Scalar, PhysicalDim, TopologyDim> local;
    for (index_t i = 0; i < TopologyDim; ++i) {
      local.col(i) = cmap(verts[cell[i]]);
    }
    const element elem(local);
    local_mat result;

    for (index_t i = 0; i < TopologyDim; ++i) {
      for (index_t j = 0; j < TopologyDim; ++j) {
        Scalar out = 0;
        if constexpr (Partial == 0) {
          out = elem.integrate_f_f(i, j);
        } else if constexpr (Partial == 1) {
          // out = elem.integrate_pf_pf(i, j, 0, 0) + elem.integrate_pf_pf(i, j, 1, 1);
          for (index_t kl = 0; kl < PhysicalDim; ++kl) {
            out += elem.integrate_pf_pf(i, j, kl, kl);
          }
        } else {
          static_assert(mp::internal::always_false_v<local_mat>, "Invalid P");
        }
        result(i, j) = out;
      }
    }

    result = result * weight * alpha_;
    cmap(dst) += result;
  }
};

template <typename Scalar, typename Device, index_t PhysicalDim, index_t TopologyDim>
linear_element_integrator<std::remove_const_t<Scalar>, Device, PhysicalDim, TopologyDim, 0> mass_integrator(
    basic_unstructured_view<Scalar, Device, PhysicalDim, TopologyDim> mesh, Scalar alpha = 1) {
  return linear_element_integrator<Scalar, Device, PhysicalDim, TopologyDim, 0>(mesh.as_const(), alpha);
}

template <typename Scalar, typename Device, index_t PhysicalDim, index_t TopologyDim>
linear_element_integrator<std::remove_const_t<Scalar>, Device, PhysicalDim, TopologyDim, 1> laplace_integrator(
    basic_unstructured_view<Scalar, Device, PhysicalDim, TopologyDim> mesh, Scalar alpha = 1) {
  return linear_element_integrator<Scalar, Device, PhysicalDim, TopologyDim, 1>(mesh.as_const(), alpha);
}

}  // namespace ssim::fem
