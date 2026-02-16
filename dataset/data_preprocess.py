# Ensure you have 'kagglehub' installed and up-to-date: pip install kagglehub
import kagglehub
import os
import pandas as pd
import torch
from torchvision import transforms

# Download the latest version of the NIH Chest X-rays dataset
path = kagglehub.dataset_download("nih-chest-xrays/data", dest_dir="dataset")

print("Path to dataset files:", path)

# Define paths
BASE_DIR = path  # Use the downloaded dataset path
WORKING_DIR = "/kaggle"
ORGANIZED_DIR = os.path.join(WORKING_DIR, "organized_dataset")

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

# Load CSV file
data_entry_df = pd.read_csv(os.path.join(BASE_DIR, "Data_Entry_2017.csv"))

# Load train/val and test lists
with open(os.path.join(BASE_DIR, "train_val_list.txt"), 'r') as f:
    train_val_files = f.read().splitlines()

with open(os.path.join(BASE_DIR, "test_list.txt"), 'r') as f:
    test_files = f.read().splitlines()

# Split train_val into train and validation
from sklearn.model_selection import train_test_split
train_files, val_files = train_test_split(train_val_files, test_size=0.2, random_state=42)
