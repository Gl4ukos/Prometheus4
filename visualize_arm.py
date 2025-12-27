import pybullet as p
import pybullet_data
import time
import numpy as np


# ---------- Collision visual cloning ----------
def spawn_collision_visuals(body_id):
    visuals = []

    for link_id in range(-1, p.getNumJoints(body_id)):
        shapes = p.getCollisionShapeData(body_id, link_id)
        if not shapes:
            continue

        for s in shapes:
            geom_type = s[2]
            dims = s[3]
            local_pos = s[5]
            local_ori = s[6]

            if geom_type == p.GEOM_BOX:
                vis_shape = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=dims,
                    rgbaColor=[1, 0, 0, 1]  # opaque for debugging
                )

            elif geom_type == p.GEOM_CYLINDER:
                vis_shape = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=dims[0],
                    length=dims[1],
                    rgbaColor=[0, 0, 1, 1]
                )
            else:
                continue

            vis_body = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=vis_shape,
                basePosition=[0, 0, 0],
                baseOrientation=[0, 0, 0, 1]
            )

            visuals.append((vis_body, link_id, local_pos, local_ori))

    return visuals


# ---------- PyBullet setup ----------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)

# Load robot WITH self-collision enabled
arm = p.loadURDF(
    "urdf/arm_4dof.urdf",
    useFixedBase=True,
    flags=p.URDF_USE_SELF_COLLISION
)


# Arm info
joint_ids = [0, 1, 2, 3]
ee_link = 4

# Initial safe pose
current_angles = np.zeros(4)


# ---------- Main loop ----------
while True:
    # --- propose motion (4 DOF ONLY) ---
    proposed_angles = np.array([
        np.sin(time.time() / 8) * 0.0,
        np.sin(time.time() / 5) * 0.0,
        np.sin(time.time() / 2) * 0.0,
        np.sin(time.time() / 1) * 0.0
    ])

    # --- TEST pose ---
    for j, a in enumerate(proposed_angles):
        p.resetJointState(arm, j, a)

    p.stepSimulation()

    # --- collision check ---
    contacts = p.getContactPoints(arm, arm)

    if len(contacts) == 0:
        print("OK")
        current_angles = proposed_angles
    else:
        print("COLLISION")
        for j, a in enumerate(current_angles):
            p.resetJointState(arm, j, a)

        # visualize contact points (debug)
        for c in contacts:
            p.addUserDebugLine(
                c[5], c[6],
                [1, 0, 0],
                lineWidth=3,
                lifeTime=0.1
            )

    # --- end effector state ---
    ee_state = p.getLinkState(arm, ee_link, computeForwardKinematics=True)
    ee_pos = ee_state[4]
    # print("EE:", ee_pos)

    time.sleep(1. / 240.)
