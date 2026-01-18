import pybullet as p
import pybullet_data
import time
import numpy as np



# ---------- PyBullet setup ----------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)

arm = p.loadURDF(
    "urdf/arm_4dof.urdf",
    useFixedBase=True,
    flags=p.URDF_USE_SELF_COLLISION
)


joint_ids = [0, 1, 2, 3]
ee_link = 4

# Initial joint angles
current_angles = np.zeros(4)
delta = 0.01  # joint increment per key press

# Key mapping (use ord for lowercase letters)
key_map = {
    ord('t'): (0, +delta),
    ord('u'): (0, -delta),
    ord('y'): (1, +delta),
    ord('i'): (1, -delta),
    ord('b'): (2, +delta),
    ord('m'): (2, -delta),
    ord('n'): (3, +delta),
    ord(','): (3, -delta)
}

# ---------- Main loop ----------
while True:
    keys = p.getKeyboardEvents()
    proposed_angles = current_angles.copy()

    for key, (joint_idx, change) in key_map.items():
        # PyBullet reports pressed keys with & p.KEY_IS_DOWN
        if key in keys and keys[key] & p.KEY_IS_DOWN:
            proposed_angles[joint_idx] += change

    # --- test pose ---
    for j, a in enumerate(proposed_angles):
        p.resetJointState(arm, j, a)
    p.stepSimulation()

    # --- collision check ---
    contacts = p.getContactPoints(arm, arm)
    if len(contacts) == 0:
        # Accept the move
        print("OK")
        current_angles = proposed_angles
    else:
        print("COLLISION")
        # Reject move
        for j, a in enumerate(current_angles):
            p.resetJointState(arm, j, a)
        # optional: show collision points
        for c in contacts:
            p.addUserDebugLine(c[5], c[6], [1, 0, 0], lineWidth=3, lifeTime=0.1)

    ee_state = p.getLinkState(arm, ee_link, computeForwardKinematics=True)
    ee_pos = ee_state[4]
    print("EE:", ee_pos)

    time.sleep(1. / 240.)
