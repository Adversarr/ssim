import numpy as np
import matplotlib.pyplot as plt
import libssim

plt.figure()
ax = plt.subplot(111, projection='3d')
# Tetrahedron
vert = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
elem = np.array([[0, 1, 2, 3]], dtype=np.int32)


solver = libssim.fem.tet_lbfgs(vert, elem, 1e-2, 1e6, 0.3, 1.0)
solver.mark_dirichlet(0, np.zeros(3, dtype=np.float64))
solver.add_gravity(np.array([0, 0, -9.81], dtype=np.float64))

# print(solver.vertices())
print(solver.deformation())
solver.step()
print(solver.deformation())
for i in range(1000):
    solver.step()
    vert = solver.vertices() + solver.deformation()
    ax.cla()
    ax.scatter(vert[0, :], vert[1, :], vert[2, :])
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    plt.pause(0.01)

# print(solver.cells())
