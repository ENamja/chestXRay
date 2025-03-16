#!/bin/bash

import numpy as np
import pandas as pd
import os
from glob import glob
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader
from PIL import Image

device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)
print(f"Running on device: {device}")

# CONSTANTS
SAMPLE_SIZE=50000
IMG_SIZE=(128,128)

all_xray_df = pd.read_csv('data/Data_Entry_2017.csv')
all_image_paths = {os.path.basename(x): x for x in glob(os.path.join('', 'data', 'images*', '*', '*.png'))}
print(f"df shape: {all_xray_df.shape}")
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df['Patient Age'] = all_xray_df['Patient Age'].map(lambda x: int(x))

all_xray_df = all_xray_df[~all_xray_df["Finding Labels"].str.contains("\|")]

labels = all_xray_df['Finding Labels'].tolist()
encoder_labels = LabelEncoder()
encoder_labels.fit(labels)
numeric_labels = encoder_labels.transform(labels)
num_classes = len(encoder_labels.classes_)
all_xray_df['Numeric Labels'] = numeric_labels

all_xray_df.sample(3)

class NewImagesDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        '''
        Args:
            dataframe (pd.dataframe): Image Paths and Labels
            transform (callable, optional): Optional transform to be applied to the images
        '''
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self) -> int:
        ''' Returns the total number of samples in the dataset '''
        return len(self.dataframe)

    def __getitem__(self, idx):
        '''
        Args: 
            idx (int): Index of the sample to retrieve
        Returns:
            image (torch.Tensor): The image as a tensor
            features (torch.Tensor): Additional features (e.g., age, gender, # of follow-ups)
            label (str): The label (e.g., diagnosis) for the image
        '''
        # Get image path and label for the given index
        image_path = self.dataframe.iloc[idx]['path']
        label = self.dataframe.iloc[idx]['Numeric Labels'] # Finding labels is the target column

        image = Image.open(image_path).convert('L') # Convert to grayscale image

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Additional features
        follow_ups = self.dataframe.iloc[idx]['Follow-up #']
        age = self.dataframe.iloc[idx]['Patient Age']
        gender = self.dataframe.iloc[idx]['Patient Gender']
        position = self.dataframe.iloc[idx]['View Position']

        # Convert gathered features to a tensor
        # Note that age and # of follow ups are stored as is and both gender and camera position is stored with binary values
        features = torch.tensor([age, follow_ups, 1.0 if gender == 'M' else 0.0, 1.0 if position == 'AP' else 0.0], dtype=torch.float32) 

        return image, features, label

def train(model, model_name, num_epochs, train_dataloader, val_dataloader, criterion, optimizer, steps_per_epoch=100, save_dir='./test_weight', patience=3):
    
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    val_imgs, val_features, val_labels = next(iter(val_dataloader))

    save_path = os.path.join(save_dir, f'{model_name}_weights.pth')
    if os.path.exists(save_path):
        print(f'The file {save_path} exists. Loading previous weights...')
        model.load_state_dict(torch.load(save_path))
    else:
        print(f'The file {save_path} not found. Creating file and saving initial weights...')
        torch.save(model.state_dict(), save_path)

    best_val_loss = float('inf')
    patience_counter = 0

    # Train loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')

        # Training Phase
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        for i, data in enumerate(train_dataloader):
            # Extract image, features, and labels
            img, features, labels = data

            if i % 50 == 0:
                print(f'i: {i}')
            
            img = img.to(device)
            features = features.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(img, features)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            running_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / steps_per_epoch
        epoch_accuracy = running_correct / total_samples
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy {epoch_accuracy:.4f}')

        # Validation Phase
        model.eval()
        with torch.no_grad():
            val_imgs = val_imgs.to(device)
            val_features = val_features.to(device)
            val_labels = val_labels.to(device)

            outputs = model(val_imgs, val_features)
            val_loss = criterion(outputs, val_labels)

            print(f'Validation Loss: {val_loss.item():.4f}')

        # Learning rate scheduler
        scheduler.step(val_loss)

        # Checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f'Validation loss improved to {val_loss.item():.5f}, saving model...')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print('Early stopping triggered')
            break
            
# Use resnet18 for now (compact version of resnet50)

class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedResNet18, self).__init__()

        # Load pretrained weights
        self.resnet18 = models.resnet18(weights=True)

        # Modify the first convolutional layer to accept 1-channel (grayscale) images
        self.resnet18.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Remove the final fully connected layer
        self.resnet18.fc = nn.Identity()

        # Define a new fully connected layer to handle the concatenated features
        # ResNet-18 outputs 512 features from the convolutional layers
        # We add 4 additional features (age, follow-ups, gender, position)
        self.fc = nn.Linear(512 + 4, num_classes)  # 512 (image features) + 4 (additional features)

    def forward(self, image, features):
        # Extract image features using ResNet-18
        image_features = self.resnet18(image)

        # Concatenate image features with additional features
        combined_features = torch.cat((image_features, features), dim=1)

        # Pass through the fully connected layer
        output = self.fc(combined_features)

        return output

# Create Image Standardizer
class StandardizePerImage:
    def __call__(self, img) -> torch.Tensor:
        '''
        Standardizes an image's pixel values and converts the image to be in the correct format for a torch.Tensor (C, H, W)
        '''
        img_array = np.array(img).astype(np.float32)

        # If the image is grayscale (2D), add a channel dimension (H, W) --> (H, W, C)
        if len(img_array.shape) == 2:
            img_array = img_array[:, :, np.newaxis]

        # Standardize the image: (x - mean) / std
        mean = np.mean(img_array, axis=(0,1), keepdims=True) # Compute mean along height and width
        std = np.std(img_array, axis=(0,1), keepdims=True) # Compute std along height and width

        std[std == 0] = 1.0 # Avoid divide by zero

        # Standardize image
        img_array = (img_array - mean) / std

        return torch.from_numpy(img_array.transpose((2, 0, 1))) # tensor images are processed as (C, H, W)
            
# Define the transform method for every image
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE), # Resize images to IMG_SIZE
    transforms.RandomHorizontalFlip(), # Randomly flip some images horizontally
    transforms.RandomAffine( # Randomly apply the changes listed
        degrees=5, # Rotate image by up to 5 degrees
        translate=(0.1,0.05), # Shift horizontally by up to 10%, up to 5% vertically
        shear=10, # Slant the image by up to 10% vertically and/or horizontally
        scale=(0.85,1.15), # Scale the image by 15% (down or upscale)
        fill=0 # Fill null pixels with 0 (black)
    ),
    StandardizePerImage()
])

# Create a train and valid dataframe
# Note: we use stratify to have an even ratio of labels in both the train df and valid df
train_df, valid_df = train_test_split(all_xray_df, test_size=0.25, random_state=1234, stratify=numeric_labels)
print('train', train_df.shape[0], 'validation', valid_df.shape[0])

train_dataset = NewImagesDataset(train_df, transform=transform)
valid_dataset = NewImagesDataset(valid_df, transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=1
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=1
)

test_loader = DataLoader(
    valid_dataset,
    batch_size=1024,
    shuffle=False,
    num_workers=1
)

model = ModifiedResNet18(len(numeric_labels))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

import time

train(model, 'resnet18', 1, train_loader, valid_loader, criterion, optimizer, steps_per_epoch=4)
