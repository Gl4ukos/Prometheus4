import pybullet as p
import pybullet_data
import time
import numpy as np

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

arm = p.loadURDF("urdf/arm_4dof.urdf", useFixedBase=True)

joint_ids = [3]


while True:
    angles = [
        np.sin(time.time()) * 1.0,
        np.sin(time.time()) * 1.0,
        np.sin(time.time()) * 1.0,
        np.sin(time.time()) * 1.0,
    ]

    for j, a in zip(joint_ids, angles):
        p.setJointMotorControl2(
            arm, j,
            p.POSITION_CONTROL,
            targetPosition=a,
            force=5
        )

    p.stepSimulation()
    time.sleep(1./240.)
