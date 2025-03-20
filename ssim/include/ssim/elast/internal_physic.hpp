#pragma once

#include <mathprim/supports/eigen_dense.hpp>

#include "ssim/defines.hpp"
namespace ssim::elast::internal {

template <typename Scalar>
SSIM_PRIMFUNC Eigen::Matrix<Scalar, 2, 2> green_strain(const Eigen::Matrix<Scalar, 2, 2>& F) {
  constexpr Scalar half = Scalar(0.5);
  return half * (F.transpose() * F - Eigen::Matrix<Scalar, 2, 2>::Identity());
}

template <typename Scalar>
SSIM_PRIMFUNC Eigen::Matrix<Scalar, 2, 2> approx_green_strain(const Eigen::Matrix<Scalar, 2, 2>& F) {
  constexpr Scalar half = Scalar(0.5);
  return half * (F + F.transpose()) - Eigen::Matrix<Scalar, 2, 2>::Identity();
}

template <typename Scalar>
SSIM_PRIMFUNC Eigen::Matrix<Scalar, 3, 3> green_strain(const Eigen::Matrix<Scalar, 3, 3>& F) {
  constexpr Scalar half = Scalar(0.5);
  return half * (F.transpose() * F - Eigen::Matrix<Scalar, 3, 3>::Identity());
}

template <typename Scalar>
SSIM_PRIMFUNC Eigen::Matrix<Scalar, 3, 3> approx_green_strain(const Eigen::Matrix<Scalar, 3, 3>& F) {
  constexpr Scalar half = Scalar(0.5);
  return half * (F + F.transpose()) - Eigen::Matrix<Scalar, 3, 3>::Identity();
}

template <typename Scalar>
SSIM_PRIMFUNC Eigen::Matrix<Scalar, 2, 2> partial_determinant(const Eigen::Matrix<Scalar, 2, 2>& F) {
  // [ f11, -f10;
  //  -f01,  f00]
  return Eigen::Matrix<Scalar, 2, 2>{{F(1, 1), -F(1, 0)},  //
                                     {-F(0, 1), F(0, 0)}};
}

template <typename Scalar>
SSIM_PRIMFUNC Eigen::Matrix<Scalar, 3, 3> partial_determinant(const Eigen::Matrix<Scalar, 3, 3>& F) {
  // [Cross[f1, f2], Cross[f2, f0], Cross[f0, f1]]
  Eigen::Matrix<Scalar, 3, 3> result;
  result.col(0) = F.col(1).cross(F.col(2));
  result.col(1) = F.col(2).cross(F.col(0));
  result.col(2) = F.col(0).cross(F.col(1));
  return result;
}

template <typename Scalar>
SSIM_PRIMFUNC void add_HJ(Eigen::Matrix<Scalar, 4, 4>& H, const Eigen::Matrix<Scalar, 2, 2>& /*F*/, Scalar scale) {
  H(3, 0) += scale;
  H(0, 3) += scale;
  H(1, 2) -= scale;
  H(2, 1) -= scale;
}

template <typename Scalar>
SSIM_PRIMFUNC Eigen::Matrix<Scalar, 3, 3> x_hat(const Eigen::Vector<Scalar, 3>& Fi) {
  Eigen::Matrix<Scalar, 3, 3> x_hat;
  x_hat << Scalar(0), -Fi(2), Fi(1), Fi(2), Scalar(0), -Fi(0), -Fi(1), Fi(0), Scalar(0);
  return x_hat;
}

template <typename Scalar>
SSIM_PRIMFUNC void add_HJ(Eigen::Matrix<Scalar, 9, 9>& H, const Eigen::Matrix<Scalar, 3, 3>& F, Scalar scale) {
  const Eigen::Matrix<Scalar, 3, 3> f0 = x_hat<Scalar>(F.col(0)) * scale;
  const Eigen::Matrix<Scalar, 3, 3> f1 = x_hat<Scalar>(F.col(1)) * scale;
  const Eigen::Matrix<Scalar, 3, 3> f2 = x_hat<Scalar>(F.col(2)) * scale;
  H.template block<3, 3>(3, 0) += f2;
  H.template block<3, 3>(0, 3) -= f2;
  H.template block<3, 3>(6, 0) -= f1;
  H.template block<3, 3>(0, 6) += f1;
  H.template block<3, 3>(6, 3) += f0;
  H.template block<3, 3>(3, 6) -= f0;
}

template <typename Scalar, typename Out>
SSIM_PRIMFUNC void compute_pfpx_2(const Eigen::Matrix<Scalar, 2, 2>& DmInv, Out& pf_px) {
  const Scalar m = DmInv(0, 0);
  const Scalar n = DmInv(0, 1);
  const Scalar p = DmInv(1, 0);
  const Scalar q = DmInv(1, 1);
  const Scalar t1 = -m - p;
  const Scalar t2 = -n - q;
  // Eigen::Matrix<Scalar, 4, 6> pf_px;
  // pf_px.setZero();
#define Fij(i, j) ((i) * 2 + (j))
#define Xij(i, j) ((j) * 2 + (i))
  pf_px(Fij(0,0), Xij(0,0)) = t1;
  pf_px(Fij(0,0), Xij(0,1)) = m;
  pf_px(Fij(0,0), Xij(0,2)) = p;
  pf_px(Fij(1,0), Xij(1,0)) = t1;
  pf_px(Fij(1,0), Xij(1,1)) = m;
  pf_px(Fij(1,0), Xij(1,2)) = p;
  pf_px(Fij(0,1), Xij(0,0)) = t2;
  pf_px(Fij(0,1), Xij(0,1)) = n;
  pf_px(Fij(0,1), Xij(0,2)) = q;
  pf_px(Fij(1,1), Xij(1,0)) = t2;
  pf_px(Fij(1,1), Xij(1,1)) = n;
  pf_px(Fij(1,1), Xij(1,2)) = q;
#undef Fij
#undef Xij
}

template <typename Scalar, typename Out>
SSIM_PRIMFUNC void compute_pfpx_3(const Eigen::Matrix<Scalar, 3, 3>& DmInv, Out& pf_px) {
  const Scalar m = DmInv(0, 0);
  const Scalar n = DmInv(0, 1);
  const Scalar o = DmInv(0, 2);
  const Scalar p = DmInv(1, 0);
  const Scalar q = DmInv(1, 1);
  const Scalar r = DmInv(1, 2);
  const Scalar s = DmInv(2, 0);
  const Scalar t = DmInv(2, 1);
  const Scalar u = DmInv(2, 2);
  const Scalar t1 = -m - p - s;
  const Scalar t2 = -n - q - t;
  const Scalar t3 = -o - r - u;
#define Fij(i, j) ((i) * 3 + (j))
#define Xij(i, j) ((j) * 3 + (i))
  pf_px(Fij(0,0), Xij(0,0)) = t1;
  pf_px(Fij(0,0), Xij(0,1)) = m;
  pf_px(Fij(0,0), Xij(0,2)) = p;
  pf_px(Fij(0,0), Xij(0,3)) = s;
  pf_px(Fij(1,0), Xij(1,0)) = t1;
  pf_px(Fij(1,0), Xij(1,1)) = m;
  pf_px(Fij(1,0), Xij(1,2)) = p;
  pf_px(Fij(1,0), Xij(1,3)) = s;
  pf_px(Fij(2,0), Xij(2,0)) = t1;
  pf_px(Fij(2,0), Xij(2,1)) = m;
  pf_px(Fij(2,0), Xij(2,2)) = p;
  pf_px(Fij(2,0), Xij(2,3)) = s;
  pf_px(Fij(0,1), Xij(0,0)) = t2;
  pf_px(Fij(0,1), Xij(0,1)) = n;
  pf_px(Fij(0,1), Xij(0,2)) = q;
  pf_px(Fij(0,1), Xij(0,3)) = t;
  pf_px(Fij(1,1), Xij(1,0)) = t2;
  pf_px(Fij(1,1), Xij(1,1)) = n;
  pf_px(Fij(1,1), Xij(1,2)) = q;
  pf_px(Fij(1,1), Xij(1,3)) = t;
  pf_px(Fij(2,1), Xij(2,0)) = t2;
  pf_px(Fij(2,1), Xij(2,1)) = n;
  pf_px(Fij(2,1), Xij(2,2)) = q;
  pf_px(Fij(2,1), Xij(2,3)) = t;
  pf_px(Fij(0,2), Xij(0,0)) = t3;
  pf_px(Fij(0,2), Xij(0,1)) = o;
  pf_px(Fij(0,2), Xij(0,2)) = r;
  pf_px(Fij(0,2), Xij(0,3)) = u;
  pf_px(Fij(1,2), Xij(1,0)) = t3;
  pf_px(Fij(1,2), Xij(1,1)) = o;
  pf_px(Fij(1,2), Xij(1,2)) = r;
  pf_px(Fij(1,2), Xij(1,3)) = u;
  pf_px(Fij(2,2), Xij(2,0)) = t3;
  pf_px(Fij(2,2), Xij(2,1)) = o;
  pf_px(Fij(2,2), Xij(2,2)) = r;
  pf_px(Fij(2,2), Xij(2,3)) = u;
#undef Fij
#undef Xij
}

}  // namespace ssim::elast::internal
