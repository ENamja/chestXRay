import numpy as np
import pandas as pd
import os
from glob import glob

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

all_xray_df = pd.read_csv('data/Data_Entry_2017.csv') # Import datafram struct
all_image_paths = {os.path.basename(x): x for x in glob(os.path.join('', 'data', 'images*', '*', '*.png'))} # Get all image paths
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get) # Map paths to corresponding image
all_xray_df['Patient Age'] = all_xray_df['Patient Age'].map(lambda x: int(x)) # Convert all ages to integers

all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', '')) # No disease --> blank

from itertools import chain
all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist()))) # Find all unique labels
all_labels = [x for x in all_labels if len(x) > 0] # Remove empty strings
print(f'All Labels ({len(all_labels)}): {all_labels}')

# Create new column for each label, value 1 if patient has disease, 0 if not
for c_label in all_labels:
    if len(c_label) > 1:
        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

# Remove all labels that have less than MIN_CASES occurances
MIN_CASES = 1000
all_labels = [c_label for c_label in all_labels if all_xray_df[c_label].sum()>MIN_CASES]
print(f'Clean Labels ({len(all_labels)})', [(c_label, int(all_xray_df[c_label].sum())) for c_label in all_labels])

# # Store the amount of labels each entry has in sample_weights. Add 0.04 to avoid dividing by zero
# sample_weights = all_xray_df['Finding Labels'].map(lambda x: len(x.split('|')) if len(x) > 0 else 0).values + 4e-2
# sample_weights /= sample_weights.sum() # Normalize
# all_xray_df = all_xray_df.sample(40000, weights=sample_weights) # Use 40,000 entries with the sample_weights, prioritizing cases with more labels

# Create a column of a vector with 1s and 0s to represent which disease an entry has
all_xray_df['disease_vec'] = all_xray_df.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])

class NewImagesDataset(Dataset):
    def __init__(self, dataframe, path_col, y_col, transform=None):
        '''
        Args:
            dataframe (pd.dataframe): Image Paths and Labels
            path_col (str): Name of column containing image paths
            y_col (str): Name of column containing labels
            transform (callable, optional): Optional transform to be applied to the images
        '''
        self.dataframe = dataframe
        self.transform = transform
        self.image_paths = dataframe[path_col].values
        self.labels = [torch.tensor(np.array(label, dtype=np.float32), dtype=torch.float32) for label in dataframe[y_col]]

    def __len__(self) -> int:
        ''' Returns the total number of samples in the dataset '''
        return len(self.dataframe)

    def __getitem__(self, idx):
        '''
        Args: 
            idx (int): Index of the sample to retrieve
        Returns:
            image (torch.Tensor): The image as a tensor
            label (str): The label (e.g., diagnosis) for the image
        '''
        # Get image path and label for the given index
        image_path = self.image_paths[idx]
        label = self.labels[idx] # Finding labels is the target column

        image = Image.open(image_path).convert('L') # Convert to grayscale image

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

def train(model, model_name, num_epochs, train_dataloader, val_dataloader, criterion, optimizer, steps_per_epoch=100, save_dir='./test_weight', patience=3):
    
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    val_x, val_y = next(iter(val_dataloader))

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
        all_predictions = []
        all_labels = []

        for i, (img, labels) in enumerate(train_dataloader):
            if i >= steps_per_epoch:
                break
            
            img = img.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(img)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = (outputs > 0.5).float() # Threshold at 0.5

            # Store predictions and labels
            all_predictions.append(predictions.cpu().detach())
            all_labels.append(labels.cpu().detach())

        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        epoch_loss = running_loss / steps_per_epoch
        epoch_accuracy = (all_predictions == all_labels).float().mean().item()
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}')

        # Validation Phase
        model.eval()
        with torch.no_grad():
            val_x = val_x.to(device)
            val_y = val_y.to(device)

            outputs = model(val_x)
            val_loss = criterion(outputs, val_y)

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
    def __init__(self, num_classes, dropout_prob=0.5):
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

        # Add dropout layer (to prevent overfitting)
        self.dropout = nn.Dropout(dropout_prob)

        # Define new final layer
        self.fc = nn.Linear(512, num_classes)  # 512 (image features)

        # Add sigmoid activation for multi-label classifications
        self.sigmoid = nn.Sigmoid()

    def forward(self, image):
        # Extract image features using ResNet-18
        image_features = self.resnet18(image)

        # Apply dropout
        image_features = self.dropout(image_features)

        # Pass through the fully connected layer
        output = self.fc(image_features)

        # Apply sigmoid activation
        output = self.sigmoid(output)

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
train_df, valid_df = train_test_split(all_xray_df, test_size=0.25, random_state=1234)
print('train', train_df.shape[0], 'validation', valid_df.shape[0])

train_dataset = NewImagesDataset(train_df, 'path', 'disease_vec', transform=transform)
valid_dataset = NewImagesDataset(valid_df, 'path', 'disease_vec', transform=transform)

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

model = ModifiedResNet18(len(all_labels))

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

import time

start_time = time.time()
train(model, 'resnet18-multi', 75, train_loader, valid_loader, criterion, optimizer, steps_per_epoch=100, patience=15)
end_time = time.time()
print(f'Total time using {device} is {end_time - start_time} seconds')