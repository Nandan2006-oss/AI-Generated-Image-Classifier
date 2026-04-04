import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# -------------------------------
# ⚙️ Device (CPU/GPU)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 📁 Load Data
# -------------------------------
fake_folder = "test/fake"
real_folder = "test/real"

X, y = [], []

def load_images(folder, label, limit=500):
    for file in os.listdir(folder)[:limit]:
        path = os.path.join(folder, file)

        image = Image.open(path).resize((128, 128)).convert("RGB")
        image_array = np.array(image)

        # (H, W, C) → (C, H, W)
        image_array = np.transpose(image_array, (2, 0, 1))

        X.append(image_array)
        y.append(label)

load_images(fake_folder, 1)
load_images(real_folder, 0)

# -------------------------------
# 🔄 Preprocessing
# -------------------------------
X = np.array(X) / 255.0
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)

print("Train:", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)

# -------------------------------
# 📦 DataLoader
# -------------------------------
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=16)

# -------------------------------
# 🧠 CNN Model
# -------------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(32 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

model = CNN().to(device)

# -------------------------------
# ⚙️ Training Setup
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 🚀 Training Loop
# -------------------------------
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

# -------------------------------
# 🧪 Testing
# -------------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print("Accuracy:", accuracy)