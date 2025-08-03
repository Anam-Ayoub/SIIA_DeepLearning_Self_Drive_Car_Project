import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# =============================
# 1. Load and preprocess data
# =============================

data = pd.read_csv('data.csv')

# Features: x, z, heading
X = data[['x', 'z', 'heading']].values.astype(np.float32)

# Actions: W, A, S, D (one-hot/multi-hot)
y = data[['w', 'a', 's', 'd']].values.astype(np.float32)

# Option: Use only one action per row (if only one key is pressed)
# Here, we use multi-label (multi-hot) since multiple keys could be pressed

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to torch tensors
torch_X_train = torch.tensor(X_train)
torch_y_train = torch.tensor(y_train)
torch_X_test = torch.tensor(X_test)
torch_y_test = torch.tensor(y_test)

# =============================
# 2. Define the model
# =============================

class MLP(nn.Module):
    def __init__(self, input_dim=3, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x):
        return self.net(x)

model = MLP()

# =============================
# 3. Training setup
# =============================

criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 64
epochs = 5000

# =============================
# 4. Training loop
# =============================

for epoch in range(epochs):
    permutation = torch.randperm(torch_X_train.size()[0])
    epoch_loss = 0
    for i in range(0, torch_X_train.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = torch_X_train[indices], torch_y_train[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# =============================
# 5. Evaluation
# =============================

with torch.no_grad():
    outputs = model(torch_X_test)
    predictions = torch.sigmoid(outputs) > 0.5
    accuracy = (predictions == torch_y_test.bool()).float().mean()
    print(f"Test accuracy (element-wise): {accuracy.item():.4f}")

# =============================
# 6. Save the model
# =============================

torch.save(model.state_dict(), 'mlp_behavior_clone.pth')
print("Model saved as mlp_behavior_clone.pth")
