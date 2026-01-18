import pybullet as p
import pybullet_data
import numpy as np
import math as m

# --- PyBullet setup ---
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)

arm = p.loadURDF(
    "urdf/arm_4dof.urdf",
    useFixedBase=True,
    flags=p.URDF_USE_SELF_COLLISION
)

joint_ids = [0, 1, 2, 3]
ee_link = 4

lower = np.array([-m.pi, -m.pi, -m.pi, -m.pi])
upper = np.array([ m.pi,  m.pi,  m.pi,  m.pi])

N = 100_000
X = []  # inputs
Y = []  # outputs

# small delta to start near target
delta = 0.1
iterations = 5

while len(X) < N:
    # --- target configuration ---
    target_joints = np.random.uniform(lower, upper)
    for j, a in enumerate(target_joints):
        p.resetJointState(arm, j, a)
    if p.getContactPoints(arm, arm):
        continue
    target_ee = p.getLinkState(arm, ee_link, computeForwardKinematics=True)[4]

    # --- starting configuration near target ---
    for _ in range(iterations):
        start_joints = target_joints + np.random.uniform(-delta, delta, size=4)
        for j, a in enumerate(start_joints):
            p.resetJointState(arm, j, a)
        if p.getContactPoints(arm, arm):
            continue
        start_ee = p.getLinkState(arm, ee_link, computeForwardKinematics=True)[4]

        # --- record data ---
        inp = np.concatenate([start_joints, start_ee, target_ee])
        X.append(inp)
        Y.append(target_joints)

X = np.array(X)
Y = np.array(Y)

np.savez("ik_dataset.npz", X=X, Y=Y)
p.disconnect()
print("Dataset saved. Shape:", X.shape, Y.shape)
