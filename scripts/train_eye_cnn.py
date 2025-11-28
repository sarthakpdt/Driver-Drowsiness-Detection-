# train_eye_cnn_with_metrics.py
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from eye_cnn import EyeCNN
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --------------------------
# Hyperparameters
# --------------------------
batch_size = 128
lr = 0.001
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Dataset & Transforms
# --------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((24, 24)),
    transforms.ToTensor()
])

full_dataset = datasets.ImageFolder("data/eyes", transform=transform)

# Split: 70% train, 15% val, 15% test
total = len(full_dataset)
train_size = int(0.7 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size
train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# --------------------------
# Model, Loss, Optimizer
# --------------------------
model = EyeCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# --------------------------
# Track losses
# --------------------------
train_losses = []
val_losses = []

# --------------------------
# Training Loop
# --------------------------
for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)

    # --------------------------
    # Validation
    # --------------------------
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (pred == labels).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# --------------------------
# Plot Loss Curve
# --------------------------
plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('EyeCNN Training and Validation Loss')
plt.legend()
plt.show()

# --------------------------
# Confusion Matrix (Validation Set)
# --------------------------
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=full_dataset.classes)
disp.plot(cmap=plt.cm.Blues)
plt.title('EyeCNN Confusion Matrix (Validation Set)')
plt.show()

# --------------------------
# Test Accuracy
# --------------------------
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, pred = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (pred == labels).sum().item()

print(f"Final Test Accuracy: {100*test_correct/test_total:.2f}%")

# --------------------------
# Save model
# --------------------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/eye_cnn.pth")
print("EyeCNN model saved.")
