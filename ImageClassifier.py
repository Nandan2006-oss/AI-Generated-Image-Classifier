import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models

# -------------------------------
# ⚙️ Device (CPU/GPU)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 🔁 Transforms (Balanced)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),   # increase back
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -------------------------------
# 📦 Custom Dataset
# -------------------------------
class ImageDataset(Dataset):
    def __init__(self, fake_dir, real_dir, transform=None, limit=1000):
        self.data = []
        self.transform = transform

        for file in os.listdir(fake_dir)[:limit]:
            self.data.append((os.path.join(fake_dir, file), 1))

        for file in os.listdir(real_dir)[:limit]:
            self.data.append((os.path.join(real_dir, file), 0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# -------------------------------
# 📁 Load Dataset
# -------------------------------
fake_folder = "test/fake"
real_folder = "test/real"

dataset = ImageDataset(fake_folder, real_folder, transform=transform)

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=16)

print("Train size:", len(train_dataset))
print("Test size :", len(test_dataset))

# -------------------------------
# 🧠 CNN Model
# -------------------------------
# -------------------------------
# 🧠 Transfer Learning Model (ResNet18)
# -------------------------------
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer (IMPORTANT)
model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(device)

# -------------------------------
# ⚙️ Training Setup
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# -------------------------------
# 🚀 Training Loop
# -------------------------------
epochs = 30

for epoch in range(epochs):
    model.train()
    total_loss = 0

    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ✅ Train accuracy
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train

    # -------------------------------
    # 🧪 Test Evaluation
    # -------------------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")

print("Training Complete!")