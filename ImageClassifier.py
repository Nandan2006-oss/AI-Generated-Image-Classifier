import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# -------------------------------
# ⚙️ Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 🔁 Transforms (FIXED)
# -------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    # ✅ ImageNet normalization (VERY IMPORTANT)
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# 📦 Custom Dataset
# -------------------------------
class ImageDataset(Dataset):
    def __init__(self, fake_dir, real_dir, transform=None):
        self.data = []
        self.transform = transform

        for file in os.listdir(fake_dir):
            self.data.append((os.path.join(fake_dir, file), 1))

        for file in os.listdir(real_dir):
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
# 📁 Dataset Paths (FIX THIS STRUCTURE)
# -------------------------------
train_fake = "dataset/train/fake"
train_real = "dataset/train/real"

test_fake = "dataset/test/fake"
test_real = "dataset/test/real"

train_dataset = ImageDataset(train_fake, train_real, transform=train_transform)
test_dataset  = ImageDataset(test_fake, test_real, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=16)

print("Train size:", len(train_dataset))
print("Test size :", len(test_dataset))

# -------------------------------
# 🧠 Transfer Learning Model (ResNet18)
# -------------------------------
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# 🔥 Unfreeze last block (LEVEL 6 move)
for name, param in model.named_parameters():
    if "layer4" in name:
        param.requires_grad = True

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(device)

# -------------------------------
# ⚙️ Training Setup
# -------------------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

# -------------------------------
# 🚀 Training Loop
# -------------------------------
epochs = 20

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

        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_acc = 100 * correct_train / total_train

    # -------------------------------
    # 🧪 Testing
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

    print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")

print("✅ Training Complete!")