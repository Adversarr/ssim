import numpy as np
import matplotlib.pyplot as plt
from pyssim.fem import TetFiniteElementSolver_Host, unit_box

plt.figure()
ax = plt.subplot(111, projection='3d')
nx = 5
vert, elem = unit_box(4 * nx, nx, nx)
vert = vert.T.copy()
vert[:, 0] *= 4
elem = elem.T.copy()


solver = TetFiniteElementSolver_Host(vert, elem, 1e-2, 1e6, 0.3, 100.0, 'lbfgs_pd')
solver.set_rtol(1e-3)
solver.mark_dirichlet_batched(np.arange(0, nx * nx, dtype=np.int32), np.zeros((3, nx * nx), dtype=np.float64))

solver.add_gravity(np.array([0, 0, -9.81], dtype=np.float64))
solver.reset()
for i in range(1000):
    dt = solver.step()
    print(f"Step {i}, dt = {dt:.3e}ms, fps = {1 / dt * 1000:.3e}")
    vert = solver.vertices() + solver.deformation()
    ax.cla()
    ax.scatter(vert[0, :], vert[1, :], vert[2, :], c='r')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    plt.pause(0.01)

# print(solver.cells())
