import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import math as m

MAX_X = 3.0
MAX_Y = 3.0
MAX_Z = 3.2

# --- Neural network ---
class IKNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 256),  # 4 joints + 3 current EE + 3 target EE
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4)    # target joint angles
        )

    def forward(self, x):
        return self.net(x)

# --- Load dataset ---
data = np.load("ik_dataset.npz")
X = data["X"]
Y = data["Y"]

# --- Normalize ---
X_norm = X.copy()
X_norm[:, 0:4] /= m.pi                # current joints
X_norm[:, 4]   /= MAX_X                # current EE x
X_norm[:, 5]   /= MAX_Y                # current EE y
X_norm[:, 6]   /= MAX_Z                # current EE z
X_norm[:, 7]   /= MAX_X                # target EE x
X_norm[:, 8]   /= MAX_Y                # target EE y
X_norm[:, 9]   /= MAX_Z                # target EE z

Y_norm = Y / m.pi  # normalize output joint angles

# --- Train / val / test split ---
N = X.shape[0]
indices = np.random.permutation(N)
train_end = int(0.7 * N)
val_end   = int(0.85 * N)

X_train, Y_train = X_norm[indices[:train_end]], Y_norm[indices[:train_end]]
X_val, Y_val     = X_norm[indices[train_end:val_end]], Y_norm[indices[train_end:val_end]]
X_test, Y_test   = X_norm[indices[val_end:]], Y_norm[indices[val_end:]]

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(Y_train, dtype=torch.float32))
val_dataset   = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                              torch.tensor(Y_val, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)

# --- Model, optimizer, loss ---
model = IKNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# --- Training loop ---
for epoch in range(200):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            val_loss += loss_fn(pred, yb).item()
    val_loss /= len(val_loader)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

torch.save(model.state_dict(), "ik_model.pth")
print("Model saved!")
