import numpy as np
import matplotlib.pyplot as plt
import libssim
from pyssim.fem import TetFiniteElementSolver_Host, unit_box

# Simulator = libssim.fem.tet_lbfgs_cuda
Simulator = libssim.fem.tet_ncg_cuda_ext_ai
plt.figure()
ax = plt.subplot(111, projection='3d')
nx = 7
vert, elem = unit_box(4 * nx, nx, nx)
vert = vert.T.copy()
vert[:, 1] -= 0.5
vert[:, 2] -= 0.5
vert[:, 0] *= 4
elem = elem.T.copy()

TIME_STEP=1e-2
solver = Simulator(vert, elem, TIME_STEP, 1e6, 0.3, 10.0)
solver.set_rtol(1e-3)

print("inited.")
solver.mark_dirichlet_batched(np.arange(0, nx * nx, dtype=np.int32), np.zeros((3, nx * nx), dtype=np.float64).copy())

x = vert[:, 0]
right_nodes = np.where(x == 4)[0]
solver.mark_dirichlet_batched(right_nodes, np.zeros((3, right_nodes.size), dtype=np.float64))

def rotate_around_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return R

solver.add_gravity(np.array([0, 0, -9.81], dtype=np.float64))
solver.reset()
world_time = 0.
print("start")
for i in range(1000):
    world_time += TIME_STEP
    if i < 300:
        vert_right_original = vert[right_nodes, :].copy()
        vert_right_curr = vert_right_original @ rotate_around_x(world_time)
        vert_right_deform = vert_right_curr - vert_right_original
        solver.mark_dirichlet_batched(right_nodes, vert_right_deform.T.copy())
    
    dt = solver.step()
    print(f"Step {i}, dt = {dt:.3e}ms, fps = {1 / dt * 1000:.3e}")
    plot_vert = vert.T + solver.deformation()
    ax.cla()
    ax.scatter(plot_vert[0, :], plot_vert[1, :], plot_vert[2, :], c='r')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    plt.pause(0.01)
plt.show()
