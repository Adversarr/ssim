#pragma once
#include <mathprim/supports/eigen_dense.hpp>

#include "ssim/mesh/basic_mesh.hpp"
namespace ssim::mesh {

template <typename Scalar>
manifold_mesh<Scalar, mp::device::cpu> simple_cube(Scalar size) {
  using S = Scalar;
  using D = mp::device::cpu;
  manifold_mesh<S, D> mesh(8, 12);
  auto vert = mp::eigen_support::cmap(mesh.vertices());
  auto elem = mp::eigen_support::cmap(mesh.cells());
  using vvec = Eigen::Vector<S, 3>;
  using ivec = Eigen::Vector<index_t, 3>;
  vert.col(0) = vvec{1, 1, 1};
  vert.col(1) = vvec{-1, 1, 1};
  vert.col(2) = vvec{-1, -1, 1};
  vert.col(3) = vvec{1, -1, 1};
  vert.col(4) = vvec{1, -1, -1};
  vert.col(5) = vvec{1, 1, -1};
  vert.col(6) = vvec{-1, 1, -1};
  vert.col(7) = vvec{-1, -1, -1};

  elem.col(0) = ivec{0, 1, 2};
  elem.col(1) = ivec{0, 2, 3};
  elem.col(2) = ivec{0, 3, 4};
  elem.col(3) = ivec{0, 4, 5};
  elem.col(4) = ivec{0, 5, 6};
  elem.col(5) = ivec{0, 6, 1};
  elem.col(6) = ivec{1, 6, 7};
  elem.col(7) = ivec{1, 7, 2};
  elem.col(8) = ivec{7, 4, 3};
  elem.col(9) = ivec{7, 3, 2};
  elem.col(10) = ivec{4, 7, 6};
  elem.col(11) = ivec{4, 6, 5};
  return mesh;
}

/**
 * @brief Generate a unit box mesh, [0, 1]^3,
 *
 * @param nx number of vertices in x direction.
 * @param ny number of vertices in y direction.
 * @param nz number of vertices in z direction.
 * @return tetmesh_3d<Scalar, mp::device::cpu>
 */
template <typename Scalar>
tet_mesh<Scalar, mp::device::cpu> unit_box(index_t nx = 2, index_t ny = 2, index_t nz = 2) {
  using S = Scalar;
  using D = mp::device::cpu;
  index_t nv = nx * ny * nz;
  index_t nt = 5 * (nx - 1) * (ny - 1) * (nz - 1);
  tet_mesh<S, D> mesh(nv, nt);
  auto vert = mp::eigen_support::cmap(mesh.vertices());  // [3, nv]
  auto elem = mp::eigen_support::cmap(mesh.cells());     // [4, nt]

  index_t id = 0;
  using vvec = Eigen::Vector<S, 3>;
  using ivec = Eigen::Vector<index_t, 4>;
  for (index_t i = 0; i < nx; ++i) {
    for (index_t j = 0; j < ny; ++j) {
      for (index_t k = 0; k < nz; ++k) {
        vert.col(id) = vvec(i / S(nx - 1), j / S(ny - 1), k / S(nz - 1));
        id++;
      }
    }
  }

  id = 0;
  for (index_t i = 0; i < nx - 1; ++i) {
    for (index_t j = 0; j < ny - 1; ++j) {
      for (index_t k = 0; k < nz - 1; ++k) {
        index_t idx000 = i * ny * nz + j * nz + k;
        index_t idx100 = (i + 1) * ny * nz + j * nz + k;
        index_t idx010 = i * ny * nz + (j + 1) * nz + k;
        index_t idx001 = i * ny * nz + j * nz + k + 1;
        index_t idx110 = (i + 1) * ny * nz + (j + 1) * nz + k;
        index_t idx101 = (i + 1) * ny * nz + j * nz + k + 1;
        index_t idx011 = i * ny * nz + (j + 1) * nz + k + 1;
        index_t idx111 = (i + 1) * ny * nz + (j + 1) * nz + k + 1;
        if (i % 2 == 1) {
          std::swap(idx000, idx100);
          std::swap(idx010, idx110);
          std::swap(idx001, idx101);
          std::swap(idx011, idx111);
        }

        if (j % 2 == 1) {
          std::swap(idx000, idx010);
          std::swap(idx100, idx110);
          std::swap(idx001, idx011);
          std::swap(idx101, idx111);
        }

        if (k % 2 == 1) {
          std::swap(idx000, idx001);
          std::swap(idx100, idx101);
          std::swap(idx010, idx011);
          std::swap(idx110, idx111);
        }

        elem.col(id++) = ivec(idx000, idx100, idx010, idx001);
        elem.col(id++) = ivec(idx010, idx100, idx110, idx111);
        elem.col(id++) = ivec(idx100, idx010, idx001, idx111);
        elem.col(id++) = ivec(idx100, idx001, idx101, idx111);
        elem.col(id++) = ivec(idx011, idx001, idx010, idx111);
      }
    }
  }

  return mesh;
}

}  // namespace ssim::mesh