import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms 

from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.utils.data import Subset
from collections import Counter
from torch.optim.lr_scheduler import StepLR
from torch.amp import GradScaler, autocast
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt




# Parameters
input_size = 224
num_classes = 122
batch_size = 32
epochs = 1
learning_rate = 0.001
device = torch.device("cuda" )
patience = 6

# Transformations for resizing and normalizing
transforms_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomPerspective(distortion_scale=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match input size of the model
    transforms.ToTensor(),          # Convert PIL image to PyTorch tensor
    transforms.Normalize(           # Normalize using ImageNet mean and std
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
# Create the dataset
test_dataset = datasets.ImageFolder(root=r"C:\Users\Mega-PC\Desktop\machine\organized_data\test", transform=val_transforms)
train_dataset = datasets.ImageFolder(root=r"C:\Users\Mega-PC\Desktop\machine\organized_data\train", transform=transforms_pipeline)
val_dataset = datasets.ImageFolder(root=r"C:\Users\Mega-PC\Desktop\machine\organized_data\validation", transform=val_transforms)

# Data Loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Get class counts
class_counts = Counter([label for _, label in train_dataset.samples])
total_samples = sum(class_counts.values())
class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
weights = torch.tensor([class_weights[i] for i in range(len(class_counts))], dtype=torch.float).to(device)



# Load EfficientNetB0
base_model = models.efficientnet_b0(pretrained=True)
for param in base_model.features.parameters():
    param.requires_grad = False

# Unfreeze the last 5 layers
for param in list(base_model.features.parameters())[-250:]:
    param.requires_grad = True

# Add custom layers
num_features = base_model.classifier[1].in_features
base_model.classifier = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes),
    
)
model = base_model.to(device)
#print(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam([
    {'params': base_model.features.parameters(), 'lr': 1e-4},  # Lower learning rate for feature extractor
    {'params': base_model.classifier.parameters(), 'lr': 1e-3}  # Higher learning rate for classifier
], weight_decay=1e-3)
scheduler = StepLR(optimizer, step_size=2, gamma=0.3)
scaler = GradScaler()

#Load the checkpoint
checkpoint = torch.load('best_checkpoint.pth')

# Restore model, optimizer, scheduler, and metadata
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1
best_val_accuracy = checkpoint['best_val_accuracy']

print(f"Resuming from epoch {start_epoch} with Best Validation Accuracy: {best_val_accuracy:.2f}%")
model
#print(val_dataset.class_to_idx)


# Training Loop
early_stop_counter = 0
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    train_loader_tqdm = tqdm(train_loader, total=len(train_loader), desc="Training", leave=True)
    for batch_idx, (images, labels) in enumerate(train_loader_tqdm):
        images, labels = images.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        with autocast(device_type="cuda", dtype=torch.float16): 
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward pass and optimize
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        train_loader_tqdm.set_postfix({
            "Batch Loss": f"{loss.item():.4f}",
            "Accuracy": f"{100 * correct / total:.2f}%"
        })
        # Monitor progress every 100 steps
        if (batch_idx + 1) % 100 == 0:
            batch_accuracy = 100 * correct / total
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}, Running Accuracy: {batch_accuracy:.2f}%")

    torch.save(model,"full_model.pth")
    train_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

    #Validation Loop
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    val_loss /= len(val_loader)
    val_accuracy = 100 * val_correct / val_total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(classification_report(y_true, y_pred))

    # Save the best model weights
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        early_stop_counter = 0
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_accuracy': best_val_accuracy
        }
        torch.save(checkpoint, 'best_checkpoint.pth')  # Save everything in one file
        print(f"Checkpoint saved with Validation Accuracy: {val_accuracy:.2f}%")
    else:
        early_stop_counter += 1
        print(f"No improvement in validation accuracy for {early_stop_counter} epochs.")

    # Early stopping check
    if early_stop_counter >= patience:
        print("Early stopping triggered!")
        break
    scheduler.step()



# # Testing Loop
# model.eval()
# test_correct = 0
# test_total = 0
# y_pred = []  # Collect predictions
# y_true = []
# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = outputs.max(1)
#         test_total += labels.size(0)
#         test_correct += predicted.eq(labels).sum().item()
#         y_pred.extend(predicted.cpu().numpy())  # Store predicted labels
#         y_true.extend(labels.cpu().numpy())    # Store true labels

# test_accuracy = 100 * test_correct / test_total
# print(f"Test Accuracy: {test_accuracy:.2f}%")
# print(f"True Labels: {labels.cpu().numpy()}")
# print(f"Predicted Labels: {predicted.cpu().numpy()}\n")
# print(classification_report(y_true, y_pred))

