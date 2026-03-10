"""
Brain Tumor Detection Using VGG (MRI Images)
Part E – Experimental Analysis (20 Marks)

Experiments:
1. Learning rate comparison: 0.001 vs 0.0001
2. Frozen vs unfrozen feature layers
3. Epoch comparison: 3, 5, and 10 epochs

Results are printed in table format and saved to experiments_results.png.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
SAVE_DIR = os.path.dirname(__file__)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
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

train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


def build_model(freeze_features=False):
    """Build a VGG16 model with optional frozen feature layers."""
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    return model.to(device)


def train_model(model, lr, num_epochs):
    """Train the model and return per-epoch losses and accuracies."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    epoch_losses = []
    epoch_accs = []

    for epoch in range(num_epochs):
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
        epoch_losses.append(epoch_loss)
        epoch_accs.append(epoch_acc)
        print(f"  Epoch [{epoch+1}/{num_epochs}]  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.2f}%")

    return epoch_losses, epoch_accs


def evaluate_model(model):
    """Evaluate the model on the test set and return accuracy."""
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
    return 100.0 * correct / total


# ============================================================
# Experiment 1: Learning Rate Comparison (0.001 vs 0.0001)
# ============================================================
print("=" * 60)
print("EXPERIMENT 1: Learning Rate Comparison")
print("=" * 60)

results_lr = {}
for lr in [0.001, 0.0001]:
    print(f"\n--- LR = {lr} ---")
    model = build_model(freeze_features=False)
    losses, accs = train_model(model, lr=lr, num_epochs=5)
    test_acc = evaluate_model(model)
    results_lr[lr] = {
        "train_acc": accs[-1],
        "test_acc": test_acc,
        "losses": losses,
    }
    print(f"  Final Train Acc: {accs[-1]:.2f}%  |  Test Acc: {test_acc:.2f}%")

# ============================================================
# Experiment 2: Frozen vs Unfrozen Feature Layers
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT 2: Frozen vs Unfrozen Layers")
print("=" * 60)

results_freeze = {}
for freeze, label in [(False, "Unfrozen (all layers)"), (True, "Frozen (features only)")]:
    print(f"\n--- {label} ---")
    model = build_model(freeze_features=freeze)
    losses, accs = train_model(model, lr=0.001, num_epochs=5)
    test_acc = evaluate_model(model)
    results_freeze[label] = {
        "train_acc": accs[-1],
        "test_acc": test_acc,
    }
    print(f"  Final Train Acc: {accs[-1]:.2f}%  |  Test Acc: {test_acc:.2f}%")

# ============================================================
# Experiment 3: Epoch Comparison (3, 5, 10)
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT 3: Epoch Comparison")
print("=" * 60)

results_epochs = {}
for num_ep in [3, 5, 10]:
    print(f"\n--- Epochs = {num_ep} ---")
    model = build_model(freeze_features=False)
    losses, accs = train_model(model, lr=0.001, num_epochs=num_ep)
    test_acc = evaluate_model(model)
    results_epochs[num_ep] = {
        "train_acc": accs[-1],
        "test_acc": test_acc,
    }
    print(f"  Final Train Acc: {accs[-1]:.2f}%  |  Test Acc: {test_acc:.2f}%")

# ============================================================
# Print Results Tables
# ============================================================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

# Table 1: Learning Rate
print("\nTable 1: Learning Rate Comparison (5 epochs)")
print(f"{'Learning Rate':<18} {'Train Acc (%)':<16} {'Test Acc (%)':<14}")
print("-" * 48)
for lr, res in results_lr.items():
    print(f"{lr:<18} {res['train_acc']:<16.2f} {res['test_acc']:<14.2f}")

# Table 2: Frozen vs Unfrozen
print("\nTable 2: Frozen vs Unfrozen Layers (lr=0.001, 5 epochs)")
print(f"{'Configuration':<30} {'Train Acc (%)':<16} {'Test Acc (%)':<14}")
print("-" * 60)
for label, res in results_freeze.items():
    print(f"{label:<30} {res['train_acc']:<16.2f} {res['test_acc']:<14.2f}")

# Table 3: Epochs
print("\nTable 3: Epoch Comparison (lr=0.001)")
print(f"{'Epochs':<10} {'Train Acc (%)':<16} {'Test Acc (%)':<14}")
print("-" * 40)
for ep, res in results_epochs.items():
    print(f"{ep:<10} {res['train_acc']:<16.2f} {res['test_acc']:<14.2f}")

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: LR comparison training loss
for lr, res in results_lr.items():
    axes[0].plot(range(1, 6), res["losses"], marker='o', label=f"LR={lr}")
axes[0].set_title("Exp 1: Training Loss by Learning Rate")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

# Plot 2: Frozen vs Unfrozen bar chart
configs = list(results_freeze.keys())
train_accs = [results_freeze[c]["train_acc"] for c in configs]
test_accs = [results_freeze[c]["test_acc"] for c in configs]
x = range(len(configs))
bar_w = 0.35
axes[1].bar([i - bar_w/2 for i in x], train_accs, bar_w, label="Train Acc")
axes[1].bar([i + bar_w/2 for i in x], test_accs, bar_w, label="Test Acc")
axes[1].set_title("Exp 2: Frozen vs Unfrozen")
axes[1].set_xticks(list(x))
axes[1].set_xticklabels(["Unfrozen", "Frozen"], fontsize=9)
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(axis='y')

# Plot 3: Epoch comparison bar chart
epochs_list = list(results_epochs.keys())
train_accs_e = [results_epochs[e]["train_acc"] for e in epochs_list]
test_accs_e = [results_epochs[e]["test_acc"] for e in epochs_list]
x2 = range(len(epochs_list))
axes[2].bar([i - bar_w/2 for i in x2], train_accs_e, bar_w, label="Train Acc")
axes[2].bar([i + bar_w/2 for i in x2], test_accs_e, bar_w, label="Test Acc")
axes[2].set_title("Exp 3: Accuracy by # Epochs")
axes[2].set_xticks(list(x2))
axes[2].set_xticklabels([str(e) for e in epochs_list])
axes[2].set_xlabel("Epochs")
axes[2].set_ylabel("Accuracy (%)")
axes[2].legend()
axes[2].grid(axis='y')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "experiments_results.png"), dpi=150)
plt.show()
print("\nExperiment plots saved to experiments_results.png")
