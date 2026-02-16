# Chest X-Ray Image Classification

# This code demonstrates how to organize and process a dataset of chest X-ray images, and how
# to set up and train a DenseNet model for image classification using PyTorch.
import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)

# Define paths
BASE_DIR = "dataset/kaggle"
WORKING_DIR = "/kaggle"
ORGANIZED_DIR = os.path.join(WORKING_DIR, "organized_dataset")

# Create organized dataset directories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(ORGANIZED_DIR, split), exist_ok=True)

# Load CSV file
data_entry_df = pd.read_csv(os.path.join(BASE_DIR, "Data_Entry_2017.csv"))

# Load train/val and test lists
with open(os.path.join(BASE_DIR, "train_val_list.txt"), 'r') as f:
    train_val_files = f.read().splitlines()

with open(os.path.join(BASE_DIR, "test_list.txt"), 'r') as f:
    test_files = f.read().splitlines()

# Split train_val into train and validation
train_files, val_files = train_test_split(train_val_files, test_size=0.2, random_state=42)

print(f"Number of training files: {len(train_files)}")
print(f"Number of validation files: {len(val_files)}")
print(f"Number of test files: {len(test_files)}")

def copy_images(file_list, split):
    """
    Copy images to their respective directories based on their primary label.

    Args:
        file_list: List of image filenames
        split: One of 'train', 'val', or 'test'
    """
    for img_file in file_list:
        # Find the image in the directory structure
        for i in range(1, 13):
            folder = f"images_{i:03d}"
            img_path = os.path.join(BASE_DIR, folder, "images", img_file)

            if os.path.exists(img_path):
                # Get label from Data_Entry_2017.csv
                labels = data_entry_df[data_entry_df['Image Index'] == img_file]['Finding Labels'].iloc[0]
                # Take first label if multiple exist
                primary_label = labels.split('|')[0]

                # Create label directory if it doesn't exist
                label_dir = os.path.join(ORGANIZED_DIR, split, primary_label)
                os.makedirs(label_dir, exist_ok=True)

                # Copy image to appropriate directory
                shutil.copy2(img_path, os.path.join(label_dir, img_file))
                break

# Organize the dataset
print("Organizing training data...")
copy_images(train_files, 'train')
print("Organizing validation data...")
copy_images(val_files, 'val')
print("Organizing test data...")
copy_images(test_files, 'test')

# Print dataset statistics
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(ORGANIZED_DIR, split)
    classes = os.listdir(split_dir)
    print(f"\n{split.capitalize()} set statistics:")
    for cls in classes:
        num_images = len(os.listdir(os.path.join(split_dir, cls)))
        print(f"{cls}: {num_images} images")

# Set up data transformations
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# Create datasets
image_datasets = {
    phase: datasets.ImageFolder(
        os.path.join(ORGANIZED_DIR, phase),
        transform=data_transforms[phase]
    )
    for phase in ['train', 'val', 'test']
}

# Create dataloaders
batch_size = 32
dataloaders = {
    phase: DataLoader(
        image_datasets[phase],
        batch_size=batch_size,
        shuffle=phase == 'train',
        num_workers=2
    )
    for phase in ['train', 'val', 'test']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

print("\nDataset sizes:")
for phase in dataset_sizes:
    print(f"{phase}: {dataset_sizes[phase]} images")

print("\nClasses:", class_names)

# Set up the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Initialize DenseNet model
model = models.densenet121(pretrained=True)
num_features = model.classifier.in_features
num_classes = len(class_names)

# Modify the classifier
model.classifier = nn.Sequential(
    nn.Linear(num_features, num_classes),
    nn.LogSoftmax(dim=1)
)

# Move model to device
model = model.to(device)

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nModel setup complete!")
print(f"Number of classes: {num_classes}")

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=10):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Train the model
model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=10)

# Save the trained model
os.makedirs("implementation/model", exist_ok=True)
torch.save(model.state_dict(), 'implementation/model/trained_model.pth')

from PIL import Image

def predict_image(image_path, model, device, class_names, transform):
    """
    Predict the class of an image using the trained model.

    Args:
        image_path: Path to the image
        model: Trained PyTorch model
        device: Device to perform computation on
        class_names: List of class names
        transform: Transformation to apply to the image
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0).to(device)
    # Set model to evaluation mode
    model.eval()
    # Disable gradient computation
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
    # Get predicted class
    predicted_class = class_names[preds[0]]
    return predicted_class

# Example usage
test_image_path = '/path/to/your/image.jpg'  # Replace with the path to your image

# Apply the same transformations as for validation
transform = data_transforms['val']

predicted_class = predict_image(test_image_path, model, device, class_names, transform)
print(f'Predicted class: {predicted_class}')
