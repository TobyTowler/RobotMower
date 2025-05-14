import json
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from polygonToMask import PolygonDataset

# Set up data loaders
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

dataset = PolygonDataset(
    img_dir="./imgs/rawImgs/", json_dir="./imgs/annotations/", transform=transform
)

train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define model (adjust based on your specific task)
# For instance, for a classification task:
num_classes = len(dataset.class_map)
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, _, class_targets in train_loader:
        inputs = inputs.to(device)
        class_targets = class_targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, class_targets)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# Save the model
torch.save(model.state_dict(), "polygon_model.pth")
