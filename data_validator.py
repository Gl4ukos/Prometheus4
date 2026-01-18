import pybullet as p
import numpy as np

p.connect(p.DIRECT)
arm = p.loadURDF("urdf/arm_4dof.urdf", useFixedBase=True)
ee_link = 4

data = np.load("ik_dataset.npz")

for i in [0, 100, 1000, 5000]:
    angles = data["Y"][i]
    pos = data["X"][i]

    for j in range(4):
        p.resetJointState(arm, j, angles[j])

    ee = p.getLinkState(arm, ee_link, computeForwardKinematics=True)[4]
    print("Error:", np.linalg.norm(np.array(ee) - pos))

p.disconnect()
