import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
import time
import warnings
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from captum.attr import LayerGradCam
from sklearn.model_selection import KFold


# Ignore warning messages
warnings.filterwarnings("ignore")

# Define the custom dataset
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.df.iloc[idx, 1] + 1  # Add 1 to the label to shift it to the range of 0, 1, 2
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def calculate_metrics(predictions, targets):
    if isinstance(predictions, int):  # Handle single values
        predictions = [predictions]
    if isinstance(targets, int):  # Handle single values
        targets = [targets]
    precision = precision_score(targets, predictions, average='weighted')
    recall = recall_score(targets, predictions, average='weighted')
    f1 = f1_score(targets, predictions, average='weighted')
    cm = confusion_matrix(targets, predictions)
    return precision, recall, f1, cm

def forward_hook(module, input, output):
    module.saved_output = output




# Define the transformation for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the directory paths for the CSV file and image folder
csv_file_path = 'augmented_labelsaugimages.csv'
image_folder_path = 'augimages'

# Define the image dataset
image_dataset = CustomDataset(csv_file_path, image_folder_path, transform=transform)

# Define the number of models, epochs, and folds for Bagging with K-fold
num_models = 3  # Change this to the desired number of models
num_epochs = 5  # Change this to the desired number of epochs
num_folds = 3  # Change this to the desired number of folds

# Create lists to store the trained models and the training/validation accuracies
models_list = []
train_losses = [[] for _ in range(num_models * num_folds)]
train_accs = [[] for _ in range(num_models * num_folds)]
val_losses = [[] for _ in range(num_models * num_folds)]
val_accs = [[] for _ in range(num_models * num_folds)]
confusion_matrices = []
epoch_runtimes = []  # Initialize the list to store epoch runtimes

# Perform K-fold cross-validation
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_idx = 1
for train_index, val_index in kfold.split(image_dataset):
    print(f"Training Fold {fold_idx}/{num_folds}")
    print("=" * 100)
    
    # Split the dataset into training and validation sets
    train_dataset = torch.utils.data.Subset(image_dataset, train_index)
    val_dataset = torch.utils.data.Subset(image_dataset, val_index)
    
    # Define the dataloaders for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # Create a new model for the current fold
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)  # 3 output classes for classification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.0007, momentum=0.9, weight_decay=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # Add this line after defining the model
    model.layer4.register_forward_hook(forward_hook)
    # Print the model's summary
    summary(model, (3, 224, 224))
    # Training and validation loop for the current model and fold
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_losses[fold_idx-1].append(running_loss / len(train_dataloader))
        train_acc = 100 * correct / total
        train_accs[fold_idx-1].append(train_acc)
        
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
    
        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                val_loss = nn.CrossEntropyLoss()(outputs, labels)
                val_running_loss += val_loss.item()
                _, val_predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (val_predicted == labels).sum().item()
        
        val_losses[fold_idx-1].append(val_running_loss / len(val_dataloader))
        val_acc = 100 * val_correct / val_total
        val_accs[fold_idx-1].append(val_acc)
        
        end_time = time.time()
        epoch_runtime = end_time - start_time
        epoch_runtimes.append(epoch_runtime)
        print(f"Training Loss: {train_losses[fold_idx-1][-1]:.4f} | Training Accuracy: {train_acc:.2f}%")
        print(f"Validation Loss: {val_losses[fold_idx-1][-1]:.4f} | Validation Accuracy: {val_acc:.2f}%")
        print(f"Epoch Runtime: {epoch_runtime:.2f} seconds")
        print("=" * 100)
        
    # Grad-CAM visualization
    if epoch == num_epochs - 1:
        layer_gradcam = LayerGradCam(model, model.layer4)
        attribution = layer_gradcam.attribute(images, target=labels)
        attribution = torch.mean(attribution, dim=1, keepdim=True)
        attribution = nn.functional.interpolate(attribution, size=(224, 224), mode='bilinear', align_corners=False)
        attribution = attribution.cpu().detach().numpy()

        # Plot Grad-CAM visualization for one image
        plt.figure()
        plt.imshow(np.transpose(images[0].cpu().detach().numpy(), (1, 2, 0)))
        plt.imshow(attribution[0][0], cmap='jet', alpha=0.5, interpolation='bilinear')
        plt.axis('off')
        plt.title('Grad-CAM Visualization')
        plt.show()
    # Save the model for the current fold
    models_list.append(model)
    
    # Evaluate the model on the validation set for the current fold
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist())
            targets.extend(labels.tolist())
    
    # Calculate and print the performance metrics for the current fold
    precision, recall, f1, cm = calculate_metrics(predictions, targets)
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("=" * 100)
    
    # Store the confusion matrix for visualization
    confusion_matrices.append(cm)
    
    # Increment the fold index
    fold_idx += 1

# Calculate the average training and validation accuracies across folds for each model
# Calculate average training accuracies for each model
avg_train_accs = [np.mean(acc) for acc in train_accs]
avg_val_accs = [np.mean(acc) for acc in val_accs]

# Print the average training and validation accuracies for each model
print("Average Training Accuracies:")
for i, acc in enumerate(avg_train_accs):
    print(f"Model {i+1}: {acc:.2f}%")

print("\nAverage Validation Accuracies:")
for i, acc in enumerate(avg_val_accs):
    print(f"Model {i+1}: {acc:.2f}%")

# Visualize the confusion matrix for each model
for i, cm in enumerate(confusion_matrices):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - Model {i+1}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()
# Plot the training and validation losses for each fold
for i in range(num_folds):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses[i], label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses[i], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - Fold {i+1}')
    plt.legend()
    plt.show()
# Print the average epoch runtime
avg_epoch_runtime = np.mean(epoch_runtimes)
print(f"Average Epoch Runtime: {avg_epoch_runtime:.2f} seconds")

# Print the summary of all folds
print("Summary of All Folds:")
for i in range(num_folds):
    print(f"\nFold {i+1}:\n")
    print(f"Training Losses: {train_losses[i]}")
    print(f"Validation Losses: {val_losses[i]}")
    print(f"Confusion Matrix:\n{confusion_matrices[i]}")
    print("=" * 80)
    print("\n")
# Define the directory path to save the model
save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)

# Save the model's state_dict
model_path = os.path.join(save_dir, 'model_state_dict.pth')
torch.save(model.state_dict(), model_path)

# Save the entire model
full_model_path = os.path.join(save_dir, 'full_model.pth')
torch.save(model, full_model_path)

print("Model weights and model saved successfully!")
