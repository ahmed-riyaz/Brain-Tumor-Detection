"""
Brain Tumor Detection Using VGG (MRI Images)
Parts A-D: Dataset Preparation, Model Implementation, Training, and Evaluation
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ============================================================
# Part A – Dataset Preparation (15 Marks)
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Transforms: Resize to 224x224, convert to tensor, normalize for VGG
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Create DataLoaders
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = train_dataset.classes  # ['normal', 'tumor']
print(f"Classes: {class_names}")
print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# ============================================================
# Part B – Model Implementation (15 Marks)
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Load pretrained VGG16
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# Replace final fully connected layer with 2 output neurons (tumor vs normal)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)

model = model.to(device)

# Print model architecture
print("\n========== VGG16 Model Architecture ==========")
print(model)
print("================================================\n")

# ============================================================
# Part C – Training (15 Marks)
# ============================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
NUM_EPOCHS = 5

train_losses = []
train_accuracies = []

print("Starting Training...")
print("-" * 50)

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]  "
          f"Loss: {epoch_loss:.4f}  "
          f"Accuracy: {epoch_acc:.2f}%")

print("-" * 50)
print("Training Complete!\n")

# Save training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(range(1, NUM_EPOCHS + 1), train_losses, marker='o')
ax1.set_title("Training Loss per Epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True)

ax2.plot(range(1, NUM_EPOCHS + 1), train_accuracies, marker='o', color='green')
ax2.set_title("Training Accuracy per Epoch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "training_curves.png"), dpi=150)
plt.show()

# ============================================================
# Part D – Evaluation (15 Marks)
# ============================================================

model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

print("Evaluating on Test Set...")

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100.0 * correct / total
print(f"\nTest Accuracy: {test_accuracy:.2f}%")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print(f"\nConfusion Matrix:\n{cm}")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix – Brain Tumor Detection (VGG16)")
plt.savefig(os.path.join(os.path.dirname(__file__), "confusion_matrix.png"), dpi=150)
plt.show()

# Interpretation
print("\n========== Results Interpretation ==========")
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives  (Normal correctly classified):  {tn}")
print(f"False Positives (Normal misclassified as Tumor): {fp}")
print(f"False Negatives (Tumor misclassified as Normal): {fn}")
print(f"True Positives  (Tumor correctly classified):   {tp}")
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("=============================================")

# Save the trained model
torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "vgg16_brain_tumor.pth"))
print("\nModel saved as vgg16_brain_tumor.pth")
