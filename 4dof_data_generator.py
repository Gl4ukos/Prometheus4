import pybullet as p
import pybullet_data
import numpy as np

# --- Setup ---
p.connect(p.DIRECT)  # no GUI = faster
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)

arm = p.loadURDF(
    "urdf/arm_4dof.urdf",
    useFixedBase=True,
    flags=p.URDF_USE_SELF_COLLISION
)

joint_ids = [0, 1, 2, 3]
ee_link = 4

lower = np.array([-3.14, -1.57, -3.14, -1.57])
upper = np.array([ 3.14,  1.57,  3.14,  1.57])

N = 100_000

X = []  # inputs: (x,y,z)
Y = []  # outputs: joint angles

for _ in range(N):
    angles = np.random.uniform(lower, upper)

    for j, a in enumerate(angles):
        p.resetJointState(arm, j, a)

    # check self-collision
    if len(p.getContactPoints(arm, arm)) > 0:
        continue

    ee_pos = p.getLinkState(arm, ee_link)[4]

    X.append(ee_pos)
    Y.append(angles)

X = np.array(X)
Y = np.array(Y)

np.savez("ik_dataset.npz", X=X, Y=Y)

p.disconnect()
