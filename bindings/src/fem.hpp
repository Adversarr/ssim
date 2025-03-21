#pragma once
#include <mathprim/supports/nanobind.hpp>

void bind_fem_host(nb::module_& fem_mod);
void bind_fem_cuda(nb::module_& fem_mod);
