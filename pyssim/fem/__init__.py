from typing import Literal, Tuple
import libssim
import numpy as np
from scipy.sparse import csr_matrix


def unit_box(nx: int, ny: int, nz: int) -> Tuple[np.ndarray, np.ndarray]:
    return libssim.fem.unit_box(nx, ny, nz)


class TetFiniteElementSolver_Host:
    def __init__(
        self,
        vertices: np.ndarray,
        elements: np.ndarray,
        time_step: float = 1e-2,
        young_modulus: float = 1e6,
        poisson_ratio: float = 0.33,
        density: float = 1e2,
        method: Literal["lbfgs", "lbfgs_pd", "newton"] = "newton",
    ):
        if method == "lbfgs":
            self.solver = libssim.fem.tet_lbfgs(
                vertices, elements, time_step, young_modulus, poisson_ratio, density
            )
        elif method == "lbfgs_pd":
            self.solver = libssim.fem.tet_lbfgs_pd(
                vertices, elements, time_step, young_modulus, poisson_ratio, density
            )
        elif method == "newton":
            self.solver = libssim.fem.tet_newton(
                vertices, elements, time_step, young_modulus, poisson_ratio, density
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def reset(self):
        self.solver.reset()

    def prepare_step(self):
        self.solver.prepare_step()

    def compute_step(self):
        self.solver.compute_step()

    def step_next(self):
        return self.solver.step_next()

    def step(self):
        return self.solver.step()

    def mark_dirichlet(self, node_idx: int, targ_deform: np.ndarray):
        self.solver.mark_dirichlet(node_idx, targ_deform)

    def mark_dirichlet_batched(self, verts: np.ndarray, targ_deform: np.ndarray):
        self.solver.mark_dirichlet_batched(verts, targ_deform)

    def set_rtol(self, rtol: float):
        self.solver.set_rtol(rtol)

    def add_gravity(self, gravity: np.ndarray):
        self.solver.add_gravity(gravity)

    def vertices(self) -> np.ndarray:
        return self.solver.vertices()

    def cells(self) -> np.ndarray:
        return self.solver.cells()

    def deformation(self) -> np.ndarray:
        return self.solver.deformation()

    def forces(self) -> np.ndarray:
        return self.solver.forces()

    def update_energy_and_gradients(self):
        self.solver.update_energy_and_gradients()

    def update_hessian(self, make_spsd=True):
        self.solver.update_hessian(make_spsd)

    def mass_matrix(self) -> csr_matrix:
        return self.solver.mass_matrix()

    def hessian(self) -> csr_matrix:
        return self.solver.hessian()

    def hessian_nonzeros(self) -> np.ndarray:
        return self.solver.hessian_nonzeros()

    def mark_general_batched(self, verts: np.ndarray):
        self.solver.mark_general_batched(verts)


class TetFiniteElementSolver_Cuda:
    def __init__(
        self,
        vertices: np.ndarray,
        elements: np.ndarray,
        time_step: float = 1e-2,
        young_modulus: float = 1e6,
        poisson_ratio: float = 0.33,
        density: float = 1e2,
        method: Literal["lbfgs", "ncg_ext_ai", "newton"] = "newton",
    ):
        if method == "lbfgs":
            self.solver = libssim.fem.tet_lbfgs_cuda(
                vertices, elements, time_step, young_modulus, poisson_ratio, density
            )
        elif method == "newton":
            self.solver = libssim.fem.tet_newton_cuda(
                vertices, elements, time_step, young_modulus, poisson_ratio, density
            )
        elif method == "ncg_ext_ai":
            self.solver = libssim.fem.tet_ncg_cuda_ext_ai(
                vertices, elements, time_step, young_modulus, poisson_ratio, density
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def reset(self):
        self.solver.reset()

    def prepare_step(self):
        self.solver.prepare_step()

    def compute_step(self):
        self.solver.compute_step()

    def step_next(self):
        return self.solver.step_next()

    def mark_dirichlet_batched(self, verts: np.ndarray, targ_deform: np.ndarray):
        self.solver.mark_dirichlet_batched(verts, targ_deform)

    def mark_general_batched(self, verts: np.ndarray):
        self.solver.mark_general_batched(verts)

    def set_rtol(self, rtol: float):
        self.solver.set_rtol(rtol)

    def add_gravity(self, gravity: np.ndarray):
        self.solver.add_gravity(gravity)

    def deformation(self) -> np.ndarray:
        return self.solver.deformation()

    def set_matrix(self, values: np.ndarray):
        """Only available when solver is tet_ncg_cuda_ext_ai

        Args:
            values: The value of compressed row of CSR matrix.
        """
        self.solver.set_matrix(values)
