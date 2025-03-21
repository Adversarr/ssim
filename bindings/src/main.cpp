#include <nanobind/nanobind.h>

#include "fem.hpp"

NB_MODULE(libssim, m) {
  auto fem = m.def_submodule("fem", "FEM submodule");
  bind_fem_host(fem);
}