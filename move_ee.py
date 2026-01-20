import torch
import torch.nn as nn
import math
import numpy as np
import pybullet as p
import pybullet_data
import time


def euclidean_dist(point1, point2):
    x = point1[0] - point2[0]
    y = point1[1] - point2[1]
    z = point1[2] - point2[2]

    diff = math.sqrt(x**2 + y**2 + z**2)
    return diff

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

DEVIATION_HARDCAP = 0.5
DEVIATION_SOFTCAP = 0.2

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

home_joints = current_joints.copy()
home_ee = current_ee.copy()

LOWER_EE = np.array([-MAX_X, -MAX_Y, -MAX_Z])
UPPER_EE = np.array([ MAX_X,  MAX_Y,  MAX_Z])


deviation =0.0
def home_arm():
    global current_joints, current_ee, target_ee

    current_joints = home_joints.copy()
    current_ee = home_ee.copy()
    target_ee = home_ee.copy()

    set_target_pos(home_ee)

    for j in joint_ids:
        p.resetJointState(arm, j, home_joints[j])

def ee_loss(angles, target_ee):
    for j in joint_ids:
        p.resetJointState(arm, j, angles[j])
    ee = np.array(p.getLinkState(arm, ee_link, True)[4])
    return np.linalg.norm(ee - target_ee)


def numerical_gradient(angles, target_ee, eps=1e-4):
    grad = np.zeros_like(angles)
    base_loss = ee_loss(angles, target_ee)

    for i in range(len(angles)):
        angles_eps = angles.copy()
        angles_eps[i] += eps
        grad[i] = (ee_loss(angles_eps, target_ee) - base_loss) / eps

    return grad

def refine_with_gd(angles, target_ee, lr=0.05, steps=3):
    refined = angles.copy()

    for _ in range(steps):
        grad = numerical_gradient(refined, target_ee)
        refined -= lr * grad

    return refined


ML_THRESHOLD = 0.02
while True:
    keys = p.getKeyboardEvents()

    # --- keyboard control ---
    for key, (i, d) in key_map.items():
        if key in keys and keys[key] & p.KEY_IS_DOWN:
            target_ee[i] = np.clip(
                target_ee[i] + d,
                LOWER_EE[i],
                UPPER_EE[i]
            )
            set_target_pos(target_ee)

    # --- state update ---
    ee_state = p.getLinkState(arm, ee_link, computeForwardKinematics=True)
    current_ee = np.array(ee_state[4])

    deviation = euclidean_dist(target_ee, current_ee)

    # --- deviation handling ---
    if deviation > DEVIATION_HARDCAP:
        print(f"EXCESSIVE DEVIATION ({deviation:.3f}) → HOMING")
        home_arm()
        time.sleep(0.5)
        continue

    elif deviation > DEVIATION_SOFTCAP:
        print(f"WARNING: HIGH DEVIATION ({deviation:.3f})")

    if deviation > ML_THRESHOLD:
        # ML prediction
        target_joints = ml_ik(current_joints, current_ee, target_ee)

        # move partially toward ML guess
        step_fraction = 0.2
        proposed_joints = current_joints + step_fraction * (target_joints - current_joints)

        # gradient descent refinement
        #refined_joints = refine_with_gd(proposed_joints, target_ee, lr=0.05, steps=2)

        current_joints = proposed_joints
        print("ML")

    else:
        print("noop")



    for j in joint_ids:
        p.resetJointState(arm, j, current_joints[j])

    p.stepSimulation()
    time.sleep(1.0 / 240.0)
