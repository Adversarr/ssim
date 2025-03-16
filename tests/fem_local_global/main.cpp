#include <gtest/gtest.h>

#include <mathprim/supports/eigen_sparse.hpp>
#include <mathprim/parallel/parallel.hpp>

#include "ssim/elast/linear.hpp"
#include "ssim/elast/stable_neohookean.hpp"
#include "ssim/finite_elements/def_grad.hpp"
#include "ssim/finite_elements/global_composer.hpp"
#include "ssim/mesh/common_shapes.hpp"

using namespace ssim;
using namespace mathprim::literal;
GTEST_TEST(gather, 1d) {
  auto line = mesh::simple_line(1.0f, 4);
  auto nV = line.num_vertices(), nE = line.num_cells();
  auto cell_values_buf = mp::make_buffer<float>(nE);
  auto cell_values = cell_values_buf.view();
  for (auto& v : cell_values) {
    v = (rand() % 100) / 100.0f;
  }

  auto seq = mp::par::seq{};
  auto composer = fem::local_global_composer<float, mp::device::cpu, 1, 2>{};

  /// stress
  auto stress_on_elements_buf = mp::make_buffer<float>(nE, 2, 1_s);
  auto stress_on_nodes_buf = mp::make_buffer<float>(nV, 1_s);
  stress_on_nodes_buf.fill_bytes(0);
  auto stress_on_elements = stress_on_elements_buf.view();
  auto stress_on_nodes = stress_on_nodes_buf.view();
  for (auto [i, j, k]: stress_on_elements.shape()) {
    stress_on_elements(i, j, k) = i * 2 + j;
  }
  
  auto gather_stress = composer.force(line.const_view(), cell_values);
  mp::sparse::basic_gather_operator<float, mp::device::cpu, 1> gather_stress_operator(
      stress_on_nodes, stress_on_elements.reshape(nE * 2, 1_s), gather_stress.desc());
  seq.run(nV, gather_stress_operator);
  auto cell = line.cells();
  std::vector<float> gt(nV, 0);
  for (index_t i = 0; i < nE; ++i) {
    for (index_t j = 0; j < 2; ++j) {
      gt[cell(i, j)] += stress_on_elements(i, j, 0) * cell_values(i);
    }
  }
  for (index_t i = 0; i < nV; ++i) {
    EXPECT_FLOAT_EQ(stress_on_nodes(i, 0), gt[i]);
  }

  /// hessian
  using hes = mp::sparse::basic_sparse_matrix<float, mp::device::cpu, mathprim::sparse::sparse_format::csr>;
  auto hessian_on_elements_buf = mp::make_buffer<float>(nE, 2_s, 2_s);
  auto hessian_on_elements = hessian_on_elements_buf.view();
  for (auto [i, j, k]: hessian_on_elements.shape()) {
    hessian_on_elements(i, j, k) = i * 4 + j * 2 + k;
  }

  // build topology.
  std::vector<mp::sparse::entry<float>> topo;
  for (index_t i = 0; i < nE; ++i) {
    for (index_t j = 0; j < 2; ++j) {
      for (index_t k = 0; k < 2; ++k) {
        auto elem_value = hessian_on_elements(i, j, k) * cell_values(i);
        topo.emplace_back(cell(i, j), cell(i, k), -elem_value);
      }
    }
  }

  hes hessian = mp::sparse::make_from_coos<float, mathprim::sparse::sparse_format::csr>(
      mp::sparse::make_from_triplets<float>(topo.begin(), topo.end(), nV, nV));
  auto gather_hessian = composer.hessian(line.const_view(), cell_values);
  mp::sparse::basic_gather_operator<float, mp::device::cpu, 0> gather_hessian_operator(
      hessian.view().values(), hessian_on_elements.flatten(), gather_hessian.desc());
  seq.run(gather_hessian_operator);

  // expect zeros
  auto expectation = hessian.view().visit([](index_t /* row */, index_t /* col */, float value) {
    EXPECT_NEAR(value, 0, 1e-6);
  });
  seq.run(expectation);
}


GTEST_TEST(gather, 3d) {
  auto line = mesh::unit_box<float>(2, 2, 2);
  auto nV = line.num_vertices(), nE = line.num_cells();
  auto cell_values_buf = mp::make_buffer<float>(nE);
  auto cell_values = cell_values_buf.view();
  for (auto& v : cell_values) {
    v = (rand() % 100) / 100.0f;
  }

  auto seq = mp::par::seq{};
  auto composer = fem::local_global_composer<float, mp::device::cpu, 3, 4>{};

  /// stress
  auto stress_on_elements_buf = mp::make_buffer<float>(nE, 4_s, 3_s);
  auto stress_on_nodes_buf = mp::make_buffer<float>(nV, 3_s);
  stress_on_nodes_buf.fill_bytes(0);
  auto stress_on_elements = stress_on_elements_buf.view();
  auto stress_on_nodes = stress_on_nodes_buf.view();
  for (auto [i, j, k]: stress_on_elements.shape()) {
    stress_on_elements(i, j, k) = i * 12 + j * 3 + k;
  }
  
  auto gather_stress = composer.force(line.const_view(), cell_values);
  mp::sparse::basic_gather_operator<float, mp::device::cpu, 1> gather_stress_operator(
      stress_on_nodes, stress_on_elements.reshape(nE * 4, 3_s), gather_stress.desc());
  seq.run(nV, gather_stress_operator);
  auto cell = line.cells();
  Eigen::Matrix3Xf gt = Eigen::Matrix3Xf::Zero(3, nV);

  for (index_t i = 0; i < nE; ++i) {
    for (index_t j = 0; j < 4; ++j) {
      for (index_t k = 0; k < 3; ++k) {
        gt(k, cell(i, j)) += stress_on_elements(i, j, k) * cell_values(i);
      }
    }
  }
  for (index_t i = 0; i < nV; ++i) {
    for (index_t j = 0; j < 3; ++j) {
      EXPECT_FLOAT_EQ(stress_on_nodes(i, j), gt(j, i));
    }
  }

  /// hessian
  using hes = mp::sparse::basic_sparse_matrix<float, mp::device::cpu, mathprim::sparse::sparse_format::csr>;
  auto hessian_on_elements_buf = mp::make_buffer<float>(nE, 12_s, 12_s);
  auto hessian_on_elements = hessian_on_elements_buf.view();
  for (auto [i, j, k]: hessian_on_elements.shape()) {
    hessian_on_elements(i, j, k) = i * 144 + j * 12 + k;
  }

  // build topology.
  std::vector<mp::sparse::entry<float>> topo;
  for (index_t i = 0; i < nE; ++i) {
    for (index_t j = 0; j < 4; ++j) {
      for (index_t k = 0; k < 4; ++k) {
        for (index_t j_dof = 0; j_dof < 3; ++j_dof) {
          for (index_t k_dof = 0; k_dof < 3; ++k_dof) {
            auto elem_value = hessian_on_elements(i, j * 3 + j_dof, k * 3 + k_dof) * cell_values(i);
            auto dof_id_ij = cell(i, j) * 3 + j_dof;
            auto dof_id_ik = cell(i, k) * 3 + k_dof;
            topo.emplace_back(dof_id_ij, dof_id_ik, -elem_value);
          }
        }
      }
    }
  }

  hes hessian = mp::sparse::make_from_coos<float, mathprim::sparse::sparse_format::csr>(
      mp::sparse::make_from_triplets<float>(topo.begin(), topo.end(), nV*3, nV*3));
  auto gather_hessian = composer.hessian(line.const_view(), cell_values);
  mp::sparse::basic_gather_operator<float, mp::device::cpu, 0> gather_hessian_operator(
      hessian.view().values(), hessian_on_elements.flatten(), gather_hessian.desc());
  seq.run(gather_hessian_operator);

  // expect zeros
  auto expectation = hessian.view().visit([](index_t /* row */, index_t /* col */, float value) {
    EXPECT_NEAR(value, 0, 1e-3);
  });
  seq.run(expectation);
}

