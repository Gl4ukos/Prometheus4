import numpy as np

L1, L2, L3 = 0.3, 0.25, 0.15 #lenghts of links


# forward kinematics for 4dof arm
# returns end effectors pose (x,y,z,pitch)
def fk_4dof(t1, t2, t3, t4):
    t23 = t2 + t3
    t234 = t2 + t3 + t4

    #planar projection
    x_prime = (
          L1 * np.cos(t2)
        + L2 * np.cos(t23)
        + L3 * np.cos(t234)
    )
    z = (
          L1 * np.sin(t2)
        + L2 * np.sin(t23)
        + L3 * np.sin(t234)
    )

    #Rotate around the base
    x = x_prime * np.cos(t1)
    y = x_prime * np.sin(t1)

    pitch = t234

    return np.array([x,y,z,pitch])

if __name__ == "__main__":
    tests = [
        (0,0,0,0),
        (np.pi/4, np.pi/6, -np.pi/6, np.pi/8),
        (np.pi/2, -np.pi/4, np.pi/2, 0)
    ]
    for t in tests:
        pos = fk_4dof(*t)
        print(f"angles: {t}")
        print(f"End effector: {pos}\n")
