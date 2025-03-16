#!/usr/bin/env python
# coding: utf-8

# ### Modules

# In[2]:


import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

import torch
import torch.nn as nn
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# In[3]:


device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)
print(f"Running on device: {device}")

# change sample size depending on model complexity
SAMPLE_SIZE = 50000

# In[4]:


all_xray_df = pd.read_csv('data/Data_Entry_2017.csv')
all_image_paths = {os.path.basename(x): x for x in glob(os.path.join('', 'data', 'images*', '*', '*.png'))}
print(f"df shape: {all_xray_df.shape}")
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df['Patient Age'] = all_xray_df['Patient Age'].map(lambda x: int(x))
all_xray_df.sample(3)


# In[5]:

# ignore other features not important (eg image size)
selected_features = ['Follow-up #', 'Patient Age', 'Patient Gender', 'View Position']
selected_label = "Finding Labels"
# ignore multi-features for now
all_xray_df = all_xray_df[~all_xray_df["Finding Labels"].str.contains("\|")]
image_paths = all_xray_df.iloc[:, -1]
#image_paths = image_paths.tolist()[:SAMPLE_SIZE]
image_paths = image_paths.tolist()

features = all_xray_df[selected_features]
# one hot encoding to handle implicit info, increase # of features
features = pd.get_dummies(features, columns=['Patient Gender', 'View Position'])
# normalize ages and number of follow ups
scaler = StandardScaler()
numeric_cols = ['Patient Age', 'Follow-up #']
features[numeric_cols] = scaler.fit_transform(features[numeric_cols])
feature_names = features.columns.tolist()

print(feature_names)
print(features)

# encode labels to numbers for model
labels = all_xray_df[selected_label].tolist()
encoder_labels = LabelEncoder()
encoder_labels.fit(labels)
numeric_labels = encoder_labels.transform(labels)

# img = plt.imread(labels.iloc[0, -1])
# plt.imshow(img, cmap="gray")

# import sys
# sys.exit(0)

# In[6]:

# extends pytorchs dataset, need __getitem__ and __len__
# consider transform/target_transform methods
class ImagesDataset(Dataset):
    def __init__(self, files, labels, encoder, transforms, mode):
        super().__init__()
        self.files = files
        self.labels = labels
        self.encoder = encoder
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        pic = Image.open(self.files[index]).convert('RGB')

        if self.mode == 'train':
            x = self.transforms(pic)
            label = self.labels[index]
            y = self.encoder.transform([label])[0]
            return x, y
        elif self.mode == 'test':
            x = self.transforms(pic)
            return x, self.files[index]


# In[7]:

# In[1]:


# train_dataloader must be torch.utils.data.DataLoader, same for val
def training(model, model_name, num_epochs, train_dataloader, val_dataloader, criterion, optimizer, save_dir='./weights'):
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    
    initial_save_path = os.path.join(save_dir, f"{model_name}_initial.pth")
    torch.save(model.state_dict(), initial_save_path)
    print(f"Saved initial weights to {initial_save_path}")
    
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        for i, data in enumerate(train_dataloader, 0):
            # Get the inputs and labels
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in val_dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate validation accuracy
        val_accuracy = 100 * correct / total
        
        # Print validation statistics
        print(f'Validation loss: {val_loss / len(val_dataloader):.3f}')
        print(f'Accuracy: {val_accuracy:.2f}%')
        
        # Save checkpoint for each epoch
        checkpoint_path = os.path.join(save_dir, f"{model_name}_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_path = os.path.join(save_dir, f"{model_name}_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {best_val_acc:.2f}%")
    
    # Save the final model
    final_path = os.path.join(save_dir, f"{model_name}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f'Finished Training. Final model saved to {final_path}')
    
    return model


# In[ ]:

# apply transformations for training
transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
])


# write pytorch dataset
train_dataset = ImagesDataset(files=image_paths,
                              labels=labels,
                              encoder=encoder_labels,
                              transforms=transforms_train,
                              mode='train')

# choose model, use pretrained model
model_name = "ResNet50"
num_classes = len(encoder_labels.classes_)
model = models.resnet50(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, num_classes)
)
# model = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)
# model.classifier = nn.Linear(model.classifier.in_features, num_classes)
# model.classifier = nn.Sequential(
#     nn.Linear(model.classifier.in_features, 512),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(512, num_classes)
# )


# load data
batch_size = 64 # base 2?
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# define params
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

# train model
num_epochs = 4
trained_model = training(
    model=model,
    model_name=model_name,
    num_epochs=num_epochs,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    criterion=criterion,
    optimizer=optimizer
)

print("\ntraining complete")


# In[ ]:




