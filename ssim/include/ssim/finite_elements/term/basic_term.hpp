#pragma once
#include "ssim/mesh/basic_mesh.hpp"

namespace ssim::fem {

template <typename Derived, typename Scalar, typename Device, index_t PhysicalDim, index_t TopologyDim>
class basic_term {
public:
  using mesh_view = basic_unstructured_view<const Scalar, Device, PhysicalDim, TopologyDim>;

private:
  mesh_view mesh_;
};

}  // namespace ssim::fem
