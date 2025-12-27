import numpy as np

data = np.load("ik_dataset.npz")

X = data["X"]   # (N, 3)
Y = data["Y"]   # (N, 4)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

print("First 5 EE positions:\n", X[:5])
print("First 5 joint angles:\n", Y[:5])
