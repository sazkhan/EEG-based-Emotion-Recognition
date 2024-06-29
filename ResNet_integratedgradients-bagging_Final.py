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
from captum.attr import IntegratedGradients

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
    precision = precision_score(targets, predictions, average='weighted')
    recall = recall_score(targets, predictions, average='weighted')
    f1 = f1_score(targets, predictions, average='weighted')
    cm = confusion_matrix(targets, predictions)
    return precision, recall, f1, cm

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

# Define the train, validation, and test datasets
train_size = int(0.6 * len(image_dataset))
val_size = int(0.2 * len(image_dataset))
test_size = len(image_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    image_dataset, [train_size, val_size, test_size]
)

# Define the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=0)

# Load the pretrained GoogLeNet model
model = models.resnet34(pretrained=True)

# Modify the classifier's final layer to match the number of classes
num_classes = 3  # Change this to the number of classes in your dataset
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Print the model's summary
summary(model, (3, 224, 224))

# Define the device for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.005)


# Define the number of models and epochs for Bagging
num_models = 3  # Change this to the desired number of models
num_epochs = 8  # Change this to the desired number of epochs


# Create lists to store the trained models and the training/validation accuracies
models_list = []
train_losses = [[] for _ in range(num_models)]
train_accs = [[] for _ in range(num_models)]
val_losses = [[] for _ in range(num_models)]
val_accs = [[] for _ in range(num_models)]
confusion_matrices = []
epoch_runtimes = []  # Initialize the list to store epoch runtimes
# Training loop for Bagging
for model_idx in range(num_models):
    print(f"Training Model {model_idx+1}/{num_models}")
    print("=" * 100)
    # Create a new model and modify the classifier's final layer
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # Define the optimizer and learning rate scheduler for the current model
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training and validation loop for the current model
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 14)
        # Record the start time of the epoch
        start_time = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            


            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

        train_loss = running_loss / len(train_dataloader)
        train_acc = (correct / total) * 100.0

        train_losses[model_idx].append(train_loss)
        train_accs[model_idx].append(train_acc)
        

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        test_predictions = []
        test_targets = []

        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                running_loss += loss.item()

                test_predictions.extend(predicted.cpu().numpy())
                test_targets.extend(labels.cpu().numpy())
                

        val_loss = running_loss / len(val_dataloader)
        val_acc = (correct / total) * 100.0

        val_losses[model_idx].append(val_loss)
        val_accs[model_idx].append(val_acc)
        # Calculate the runtime of the epoch
        epoch_runtime = time.time() - start_time
        epoch_runtimes.append(epoch_runtime)
        scheduler.step()
        # Save the model's state dictionary instead of the model itself
        models_list.append(model.state_dict())

        print(f"Runtime: {epoch_runtimes[epoch]:.2f} seconds - [Train Loss: {train_loss:.4f} & Acc: {train_acc:.2f}%] - [Val Loss: {val_loss:.4f} & Acc: {val_acc:.2f}%]")
        cm = confusion_matrix(test_targets, test_predictions)
        confusion_matrices.append(cm)
    # Create an instance of the IntegratedGradients method
    integrated_gradients = IntegratedGradients(model)

    # Choose a random image from the test dataset
    image, label = test_dataset[random.randint(0, len(test_dataset))]
    image = image.unsqueeze(0)  # Add batch dimension

    # Convert the label to a PyTorch tensor
    label_tensor = torch.tensor(label)

    # Calculate the attributions using Integrated Gradients
    attributions = integrated_gradients.attribute(image, target=label_tensor)

    # Normalize the attributions to the range [0, 1]
    attributions /= torch.max(torch.abs(attributions))

    # Convert the tensor to a numpy array and transpose it
    attributions = attributions.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    # Plot the original image and its attributions
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
    plt.axis('off')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(attributions, cmap='jet', alpha=0.8)
    plt.axis('off')
    plt.title('Attributions (Integrated Gradients)')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
        
        
    # Display the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - Model {model_idx+1}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # Save the best model weights based on validation accuracy
    best_model_weights = model.state_dict()
    models_list.append(best_model_weights)

   
    print("-----------------------------")
     # Calculate and print the performance metrics on the test set
    precision, recall, f1, cm = calculate_metrics(test_predictions, test_targets)
    print("Test Set Performance Metrics")
    print("-" * 27)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

# Function to calculate performance metrics
def calculate_metrics(predictions, targets, losses):
    accuracy = (np.array(predictions) == np.array(targets)).mean() * 100.0
    precision = precision_score(targets, predictions, average='weighted')
    recall = recall_score(targets, predictions, average='weighted')
    f1 = f1_score(targets, predictions, average='weighted')
    cm = confusion_matrix(targets, predictions)
    avg_loss = np.mean(losses)
    return accuracy, precision, recall, f1, cm, avg_loss



# Evaluate the test set using the trained models
# Evaluate the test set using the trained models
test_predictions = []
test_targets = []
test_losses = []

for model_weights in models_list:
    model.load_state_dict(model_weights)
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(test_dataloader)
    test_losses.append(avg_loss)


# Calculate the performance metrics on the test set
accuracy, precision, recall, f1, cm, avg_loss = calculate_metrics(test_predictions, test_targets, test_losses)

# Print the performance metrics on the test set
print("Test Set Performance Metrics")
print("-" * 27)
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Confusion Matrix:\n{cm}")
print(f"Average Test Loss: {avg_loss:.4f}")
      
# Plot the training and validation losses for each model
plt.figure(figsize=(12, 6))
for i in range(num_models):
    plt.plot(train_losses[i], label=f"Model {i+1} Train Loss")
    plt.plot(val_losses[i], label=f"Model {i+1} Val Loss")
plt.title("Training and Validation Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss (%)")
plt.legend()
plt.show()
# Plot the training and validation accuracies for each model
plt.figure(figsize=(12, 6))
for i in range(num_models):
    plt.plot(train_accs[i], label=f"Model {i+1} Train Acc")
    plt.plot(val_accs[i], label=f"Model {i+1} Val Acc")
plt.title("Training and Validation Accuracies")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()

# Print summary of all models
print("Summary of Models:")
for model_idx in range(num_models):
    print(f"\nModel {model_idx+1}:\n")
    print(f"Training Losses: {train_losses[model_idx]}")
    print(f"Training Accuracies: {train_accs[model_idx]}")
    print(f"Validation Losses: {val_losses[model_idx]}")
    print(f"Validation Accuracies: {val_accs[model_idx]}")
    print(f"Confusion Matrix:\n{confusion_matrices[model_idx]}")
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