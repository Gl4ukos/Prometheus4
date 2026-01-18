import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

data = np.load("ik_dataset.npz")

X = data["X"]   # (N, 10)
Y = data["Y"]   # (N, 4)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

print(" X :\n", X[:15])
print(" Y :\n", Y[:15])

