import numpy as np
import matplotlib.pyplot as plt
from fk_4dof import fk_4dof

xs, ys, zs = [], [], []

for _ in range(2000):
    t1 = np.random.uniform(-np.pi, np.pi)
    t2 = np.random.uniform(-np.pi/2, np.pi/2)
    t3 = np.random.uniform(-np.pi, np.pi)
    t4 = np.random.uniform(-np.pi/2, np.pi/2)

    x, y, z, _ = fk_4dof(t1, t2, t3, t4)
    xs.append(x)
    ys.append(y)
    zs.append(z)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(xs, ys, zs, s=1)
ax.set_title("4-DOF Arm Workspace")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
