import torch
import torch.nn as nn
import math
import numpy as np
import pybullet as p
import pybullet_data
import time

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

MAX_X = 3.0
MAX_Y = 3.0
MAX_Z = 3.2

# --- EE target marker ---
sphere_radius = 0.03
visual_shape_id = p.createVisualShape(
    shapeType=p.GEOM_SPHERE,
    radius=sphere_radius,
    rgbaColor=[1, 0, 0, 1]
)
target_marker = p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=visual_shape_id,
    basePosition=[0, 0, 0]
)
def set_target_pos(pos):
    p.resetBasePositionAndOrientation(target_marker, pos, [0, 0, 0, 1])

# --- Key mapping ---
delta = 0.01
key_map = {
    ord('u'): (0, +delta),
    ord('o'): (0, -delta),
    ord('j'): (1, +delta),
    ord('l'): (1, -delta),
    ord('i'): (2, +delta),
    ord('k'): (2, -delta)
}

# --- Load model ---
class IKNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
    def forward(self, x):
        return self.net(x)

model = IKNet()
model.load_state_dict(torch.load("ik_model.pth", map_location="cpu"))
model.eval()

# --- Iterative ML IK ---
def ml_ik(current_joints, current_ee, target_ee):
    x = np.concatenate([
        current_joints / math.pi,
        current_ee / [MAX_X, MAX_Y, MAX_Z],
        target_ee / [MAX_X, MAX_Y, MAX_Z]
    ])
    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        pred_norm = model(x_tensor).numpy()
    return pred_norm * math.pi  # de-normalize

# --- Main loop ---
current_joints = np.zeros(4)
ee_state = p.getLinkState(arm, ee_link, computeForwardKinematics=True)
current_ee = np.array(ee_state[4])
target_ee = current_ee.copy()
set_target_pos(target_ee)

LOWER_EE = np.array([-MAX_X, -MAX_Y, -MAX_Z])
UPPER_EE = np.array([ MAX_X,  MAX_Y,  MAX_Z])

while True:
    keys = p.getKeyboardEvents()
    for key, (i, d) in key_map.items():
        if key in keys and keys[key] & p.KEY_IS_DOWN:
            target_ee[i] = np.clip(
                target_ee[i] + d,
                LOWER_EE[i],
                UPPER_EE[i]
            )
            set_target_pos(target_ee)

    # get current EE
    ee_state = p.getLinkState(arm, ee_link, computeForwardKinematics=True)
    current_ee = np.array(ee_state[4])

    # --- ML IK ---
    target_joints = ml_ik(current_joints, current_ee, target_ee)

    # apply joints iteratively
    step_fraction = 0.2  # apply fraction of delta for stability
    for j in range(4):
        current_joints[j] += step_fraction * (target_joints[j] - current_joints[j])
        p.resetJointState(arm, j, current_joints[j])

    p.stepSimulation()
    time.sleep(1./240.)
