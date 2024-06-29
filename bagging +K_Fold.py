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

def plot_image_with_gradcam(image, attribution, target_label):
    image = np.transpose(image, (1, 2, 0))  # Transpose the image shape
    plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Original Image')
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(image)
    ax2.imshow(attribution, cmap='jet', alpha=0.5, interpolation='bilinear')
    ax2.axis('off')
    ax2.set_title(f'Grad-CAM Visualization - Predicted: {target_label}')
    plt.show()


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
num_epochs = 7  # Change this to the desired number of epochs
num_folds = 3  # Change this to the desired number of folds

# Create lists to store the trained models and the training/validation accuracies
models_list = []
train_losses = []
train_accs = []
val_losses = []
val_accs = []
confusion_matrices = []
epoch_runtimes = []

# Perform K-fold cross-validation for each model
for model_idx in range(num_models):
    print(f"Model {model_idx+1}")
    print("=" * 100)
    
    # Create lists to store the metrics for the current model
    model_train_losses = []
    model_train_accs = []
    model_val_losses = []
    model_val_accs = []
    model_confusion_matrices = []
    model_epoch_runtimes = []
    
    # Perform K-fold cross-validation
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_idx = 1
    
    for train_index, val_index in kfold.split(image_dataset):
        print(f"Training Fold {fold_idx}/{num_folds}")
        print("-" * 100)
        
        # Split the dataset into training and validation sets
        train_dataset = torch.utils.data.Subset(image_dataset, train_index)
        val_dataset = torch.utils.data.Subset(image_dataset, val_index)
        
        # Define the dataloaders for training and validation
        train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=6, shuffle=True, num_workers=0)
        
        # Create a new model for the current fold
        model = models.googlenet(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)  # 3 output classes for classification
        
        # Print the model's summary
        summary(model, (3, 224, 224))
        
        # Move the model to the appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Define the optimizer and scheduler
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.000)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.0)
        
        # Register the forward hook for Grad-CAM
        model.inception5b.register_forward_hook(forward_hook)
        
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
            
            epoch_train_loss = running_loss / len(train_dataloader)
            epoch_train_acc = 100 * correct / total
            model_train_losses.append(epoch_train_loss)
            model_train_accs.append(epoch_train_acc)
            
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
            
            epoch_val_loss = val_running_loss / len(val_dataloader)
            epoch_val_acc = 100 * val_correct / val_total
            model_val_losses.append(epoch_val_loss)
            model_val_accs.append(epoch_val_acc)
            
            end_time = time.time()
            epoch_runtime = end_time - start_time
            model_epoch_runtimes.append(epoch_runtime)
            print(f"Training Loss: {epoch_train_loss:.4f} | Training Accuracy: {epoch_train_acc:.2f}%")
            print(f"Validation Loss: {epoch_val_loss:.4f} | Validation Accuracy: {epoch_val_acc:.2f}%")
            print(f"Epoch Runtime: {epoch_runtime:.2f} seconds")
            print("-" * 100)
        
        # Grad-CAM visualization
        if epoch == num_epochs - 1:
            layer_gradcam = LayerGradCam(model, model.inception5b)
            attribution = layer_gradcam.attribute(images, target=labels)
            attribution = torch.mean(attribution, dim=1, keepdim=True)
            attribution = nn.functional.interpolate(attribution, size=(224, 224), mode='bilinear', align_corners=False)
            attribution = attribution.cpu().detach().numpy()
    
            # Plot Grad-CAM visualization for one image
            for i in range(len(images)):
                plot_image_with_gradcam(images[i], attribution[i][0], labels[i])
        
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
        print("-" * 100)
        
        # Store the confusion matrix for visualization
        model_confusion_matrices.append(cm)
        
        # Store the epoch runtimes
        epoch_runtimes.append(model_epoch_runtimes)
        
        fold_idx += 1
    
    # Calculate the average training and validation accuracies across folds for the current model
    model_avg_train_acc = np.mean(model_train_accs)
    model_avg_val_acc = np.mean(model_val_accs)
    print(f"Average Training Accuracy: {model_avg_train_acc:.2f}%")
    print(f"Average Validation Accuracy: {model_avg_val_acc:.2f}%")
    print("=" * 100)
    
    # Store the training and validation metrics for the current model
    train_losses.append(model_train_losses)
    train_accs.append(model_train_accs)
    val_losses.append(model_val_losses)
    val_accs.append(model_val_accs)
    confusion_matrices.append(model_confusion_matrices)

# Calculate the average training and validation accuracies across models and folds
avg_train_accs = np.mean(train_accs, axis=1)
avg_val_accs = np.mean(val_accs, axis=1)

# Print the average training and validation accuracies for each model
print("Average Training Accuracies:")
for i, acc in enumerate(avg_train_accs):
    print(f"Model {i+1}: {acc:.2f}%")

print("\nAverage Validation Accuracies:")
for i, acc in enumerate(avg_val_accs):
    print(f"Model {i+1}: {acc:.2f}%")

# Rest of the code for ensemble bagging, visualization, and saving the models goes here...
# ...
# Ensemble Bagging
ensemble_predictions = []
ensemble_targets = []

with torch.no_grad():
    for images, labels in val_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        ensemble_outputs = torch.zeros(labels.size(0), 3).to(device)  # Initialize ensemble predictions
        for model in models_list:
            model.eval()
            outputs = model(images)
            ensemble_outputs += outputs
        ensemble_outputs /= len(models_list)  # Average ensemble predictions
        _, ensemble_predicted = torch.max(ensemble_outputs.data, 1)
        ensemble_predictions.extend(ensemble_predicted.tolist())
        ensemble_targets.extend(labels.tolist())

# Calculate and print the performance metrics for the ensemble
precision, recall, f1, cm = calculate_metrics(ensemble_predictions, ensemble_targets)
ensemble_accuracy = (cm[0, 0] + cm[1, 1] + cm[2, 2]) / np.sum(cm)
print(f"Ensemble Accuracy: {ensemble_accuracy*100:.2f}%")
print(f"Ensemble Precision: {precision:.2f}%")
print(f"Ensemble Recall: {recall:.2f}%")
print(f"Ensemble F1 Score: {f1:.2f}%")
print(f"Ensemble Confusion_Matrix:\n {cm}")

# Visualize the confusion matrix for each model and fold
for model_idx, model_matrices in enumerate(confusion_matrices):
    for fold_idx, cm in enumerate(model_matrices):
        vmin = np.min(cm)
        vmax = np.max(cm)
        off_diag_mask = np.eye(*cm.shape, dtype=bool)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', mask=~off_diag_mask, cmap='Blues', vmin=vmin, vmax=vmax)
        sns.heatmap(cm, annot=True, mask=off_diag_mask, cmap='OrRd', vmin=vmin, vmax=vmax, cbar_kws=dict(ticks=[]))
        plt.title(f"Confusion Matrix - Model {model_idx+1} - Fold {fold_idx+1}")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.show()


# Plot the training and validation losses for each model
for i in range(num_models):
    model_train_losses = train_losses[i]  # Training losses for the current model
    model_val_losses = val_losses[i]  # Validation losses for the current model
    
    # Ensure that the number of epochs is consistent for training and validation losses
    num_epochs = len(model_train_losses)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), model_train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), model_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - Model {i+1}')
    plt.legend()
    plt.show()


# Print the average epoch runtime
avg_epoch_runtime = np.mean(epoch_runtimes)
print(f"\nAverage Epoch Runtime: {avg_epoch_runtime:.2f} seconds")

# Print the summary of all models and folds
print("Summary of All Models and Folds:")
for i in range(num_models):
    print(f"\nModel {i+1}:")
    for j in range(num_folds):
        print(f"\nFold {j+1}:\n")
        print(f"Training Losses: {train_losses[i][j]}")
        print(f"Validation Losses: {val_losses[i][j]}")
        print(f"Confusion Matrix:\n{confusion_matrices[i][j]}")
        print("=" * 80)
        print("\n")

# Define the directory path to save the models
save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)

# Save each model's state_dict
for i, model in enumerate(models_list):
    model_path = os.path.join(save_dir, f'model_state_dict_{i+1}.pth')
    torch.save(model.state_dict(), model_path)

# Save the entire ensemble of models
ensemble_model_path = os.path.join(save_dir, 'ensemble_model.pth')
torch.save(models_list, ensemble_model_path)

print("Models and ensemble model saved successfully!")



