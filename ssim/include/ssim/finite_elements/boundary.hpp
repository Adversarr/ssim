#pragma once
#include "ssim/defines.hpp"
#include "ssim/finite_elements/defines.hpp"
#include "ssim/mesh/basic_mesh.hpp"

namespace ssim::fem {

enum class node_boundary_type : char {
  general,
  dirichlet,
};

template <typename Scalar, typename Device, index_t PhysicalDim, index_t TopologyDim, index_t DofsPerNode>
class boundary_condition {
public:
  using scalar_type = Scalar;
  using device_type = Device;
  static constexpr index_t physical_dim = PhysicalDim;
  static constexpr index_t dofs_per_node = DofsPerNode;
  using mesh_view = basic_unstructured_view<const Scalar, Device, PhysicalDim, TopologyDim>;
  using vertex_type = typename mesh_view::vertex_type;
  using cell_type = typename mesh_view::cell_type;
  using value_type = mp::contiguous_view<Scalar, mp::shape_t<DofsPerNode>, Device>;
  using const_value = mp::contiguous_view<const Scalar, mp::shape_t<DofsPerNode>, Device>;

  using dof_type_type = mp::contiguous_view<const node_boundary_type, mp::shape_t<DofsPerNode>, Device>;
  using const_batched_dof_type = mp::batched<dof_type_type>;

  using batched_value = mp::batched<value_type>;
  using const_batched_value = mp::batched<const_value>;
  using hessian_type = sparse_view<Scalar, Device>;

  struct set_values_impl {
    SSIM_PRIMFUNC void operator()(const const_value& dbc_val,     //
                                  const dof_type_type& dof_type,  //
                                  const value_type& value) const noexcept {
      for (index_t i = 0; i < DofsPerNode; ++i) {
        if (dof_type[i] == node_boundary_type::dirichlet) {
          value[i] = dbc_val[i];
        }
      }
    }
  };

  struct zero_grads_impl {
    SSIM_PRIMFUNC void operator()(const vertex_type& grads,  //
                                  const dof_type_type& dof_type) const noexcept {
      for (index_t i = 0; i < DofsPerNode; ++i) {
        if (dof_type(i) == node_boundary_type::dirichlet) {
          grads(i) = 0;
        }
      }
    }
  };

  struct identity_hessian_impl {
    const_batched_dof_type dof_type_;
    SSIM_PRIMFUNC void operator()(index_t i, index_t j, Scalar val) {
      auto itype = dof_type_(mp::ind2sub(dof_type_.shape(), i));
      auto jtype = dof_type_(mp::ind2sub(dof_type_.shape(), j));
      if (itype == node_boundary_type::dirichlet && jtype == node_boundary_type::dirichlet) {
        val = i == j ? 1 : 0;
      }
    }
  };

  boundary_condition() = default;
  SSIM_PRIMFUNC boundary_condition(const mesh_view& mesh,                    //
                                   const const_batched_dof_type& node_type,  //
                                   const const_batched_value& value) noexcept :
      mesh_(mesh), node_type_(node_type), value_(value) {}
  SSIM_INTERNAL_ENABLE_ALL_CTOR(boundary_condition);

  SSIM_PRIMFUNC const mesh_view& mesh() const noexcept { return mesh_; }
  SSIM_PRIMFUNC const_batched_dof_type node_type() const noexcept { return node_type_; }
  SSIM_PRIMFUNC const batched_value& value() const noexcept { return value_; }

  /// @brief Set the value of nodes with dirichlet boundaries.
  template <typename ParImpl>
  void value(const mp::par::parfor<ParImpl>& pf, batched_value values) {
    auto set_values = set_values_impl{};
    pf.vmap(set_values, value_, node_type_, values);
  }

  /// @brief Set the grads of dirichlet boundaries to zero.
  template <typename ParImpl>
  void grads(const mp::par::parfor<ParImpl>& pf, batched_value grads) {
    auto zero_grads = zero_grads_impl{};
    pf.vmap(zero_grads, grads, node_type_);
  }

  /// @brief Set the hessian of dirichlet boundaries to identity.
  template <typename ParImpl>
  void hessian(const mp::par::parfor<ParImpl>& pf, hessian_type hessian) {
    identity_hessian_impl ijk_task{node_type_};
    auto visit = hessian.visit(ijk_task);
    pf.run(visit);
  }

private:
  mesh_view mesh_;
  const_batched_dof_type node_type_;
  const_batched_value value_;
};

}  // namespace ssim::fem
