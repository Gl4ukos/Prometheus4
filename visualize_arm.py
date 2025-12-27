import pybullet as p
import pybullet_data
import time
import numpy as np

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
                vis = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=dims,
                    rgbaColor=[1, 0, 0, 0.3]
                )

            elif geom_type == p.GEOM_CYLINDER:
                vis = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=dims[0],
                    length=dims[1],
                    rgbaColor=[0, 0, 1, 0.3]
                )
            else:
                continue

            mb = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=vis,
                basePosition=[0, 0, 0],
                baseOrientation=[0, 0, 0, 1]
            )

            visuals.append((mb, link_id, local_pos, local_ori))

    return visuals



p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

arm = p.loadURDF("urdf/arm_4dof.urdf", useFixedBase=True)
collision_visuals = spawn_collision_visuals(arm)

ee_link = 4
p.setGravity(0, 0, 0)

while True:
    proposed_angles = [
        np.sin(time.time()/8) * 1.0,
        np.sin(time.time()/5) * 0.7,
        np.sin(time.time()/2) * 2.0,
        np.sin(time.time()/1) * 3.0

    ]

    for vis, link_id, local_pos, local_ori in collision_visuals:
        if link_id == -1:
            pos, ori = p.getBasePositionAndOrientation(arm)
        else:
            pos, ori = p.getLinkState(arm, link_id)[4:6]

        world_pos, world_ori = p.multiplyTransforms(
            pos, ori,
            local_pos, local_ori
        )

        p.resetBasePositionAndOrientation(vis, world_pos, world_ori)


    for j, a in enumerate(proposed_angles):
        p.resetJointState(arm, j, a)

    p.stepSimulation()

    # --- Collision check ---
    contacts = p.getContactPoints(bodyA=arm, bodyB=arm)
    if len(contacts) == 0:
        current_angles = proposed_angles
    else:
        for j, a in enumerate(current_angles):
            p.resetJointState(arm, j, a)


    ee_state = p.getLinkState(
        arm,
        ee_link,
        computeForwardKinematics=True
    )

    ee_pos = ee_state[4]   # world position (x, y, z)
    ee_ori = ee_state[5]   # world orientation (quaternion)

    print(f"EE pos: {ee_pos}")
    time.sleep(1./240.)

    