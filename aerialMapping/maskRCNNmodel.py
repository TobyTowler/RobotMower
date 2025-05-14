import torchvision
import os
import json
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
import os
import json
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import random


class GolfCourseDataset(Dataset):
    def __init__(self, img_dir, json_dir, transforms=None):
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.transforms = transforms
        self.json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

        # Create class mapping
        self.class_names = [
            "background",
            "green",
            "fairway",
            "bunker",
            "rough",
            "water",
        ]
        self.class_map = {
            "background": 0,
            "green": 1,
            "fairway": 2,
            "bunker": 3,
            "rough": 4,
            "water": 5,
        }

        # First pass to build class map
        for json_file in self.json_files:
            json_path = os.path.join(json_dir, json_file)
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                for shape in data.get("shapes", []):
                    if shape.get("shape_type") == "polygon":
                        label = shape.get("label", "unknown")
                        if label not in self.class_map:
                            self.class_map[label] = len(self.class_map)
                            self.class_names.append(label)
            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                print(f"Error processing {json_file}: {e}")
                continue

        # Build valid samples list
        self.samples = []
        for json_file in self.json_files:
            json_path = os.path.join(json_dir, json_file)
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                img_file = data.get("imagePath", json_file.replace(".json", ".jpg"))
                img_path = os.path.join(img_dir, os.path.basename(img_file))

                if os.path.exists(img_path):
                    self.samples.append((img_path, json_path))
            except Exception as e:
                print(f"Error processing {json_file}: {e}")

        print(
            f"Loaded {len(self.samples)} samples with {len(self.class_names)} classes"
        )
        print(f"Class mapping: {self.class_map}")

    def __len__(self):
        return len(self.samples)

    def polygon_to_mask(self, polygon, img_height, img_width):
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        # Convert polygon to numpy array
        polygon = np.array(polygon, dtype=np.int32).reshape((-1, 2))
        # Fill polygon
        cv2.fillPoly(mask, [polygon], 1)
        return mask

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        # Load annotations
        with open(json_path, "r") as f:
            data = json.load(f)

        # Initialize lists for annotations
        boxes = []
        masks = []
        labels = []

        # Process each shape
        for shape in data.get("shapes", []):
            if shape.get("shape_type") == "polygon":
                points = shape.get("points", [])
                if len(points) >= 3:  # Need at least 3 points for a polygon
                    label = shape.get("label", "unknown")
                    if label in self.class_map:
                        # Create mask
                        mask = self.polygon_to_mask(points, height, width)

                        # Find bounding box (required by Mask R-CNN)
                        pos = np.where(mask)
                        if len(pos[0]) > 0 and len(pos[1]) > 0:
                            xmin = np.min(pos[1])
                            xmax = np.max(pos[1])
                            ymin = np.min(pos[0])
                            ymax = np.max(pos[0])

                            # Prevent degenerate boxes
                            if xmax > xmin and ymax > ymin:
                                boxes.append([xmin, ymin, xmax, ymax])
                                masks.append(mask)
                                labels.append(self.class_map[label])

        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        else:
            # Return empty tensors if no annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            masks = torch.zeros((0, height, width), dtype=torch.uint8)

        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            if len(boxes) > 0
            else torch.zeros(0),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        # Apply transformations
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


# Custom transforms for both image and target
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        # Convert PIL image to tensor
        image = torchvision.transforms.functional.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # Flip image
            image = torchvision.transforms.functional.hflip(image)

            # Flip boxes
            h, w = image.shape[-2:]
            if target["boxes"].shape[0] > 0:
                target["boxes"][:, [0, 2]] = w - target["boxes"][:, [2, 0]]

            # Flip masks
            if "masks" in target:
                target["masks"] = torch.flip(target["masks"], [2])

        return image, target


def get_transform(train):
    transforms = []
    # Convert PIL image to tensor
    transforms.append(ToTensor())
    if train:
        # Add data augmentation for training
        transforms.append(RandomHorizontalFlip(0.5))

    return Compose(transforms)


# Create Mask R-CNN model
def get_model(num_classes):
    # Load pre-trained model

    model = maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace the classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


# Training function
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(data_loader)


# Main training script
def main():
    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"DEVICE {torch.get_device}")

    # Create dataset
    dataset = GolfCourseDataset(
        img_dir="./imgs/rawImgs/",
        json_dir="./imgs/annotations/",
        transforms=get_transform(train=True),
    )

    # Split into train/validation
    indices = torch.randperm(len(dataset)).tolist()
    train_size = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, indices[:train_size])
    val_dataset = torch.utils.data.Subset(
        GolfCourseDataset(
            img_dir="./imgs/rawImgs/",
            json_dir="./imgs/annotations/",
            transforms=get_transform(train=False),
        ),
        indices[train_size:],
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),  # Custom collate to handle variable sizes
    )

    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x))
    )

    # Create model
    num_classes = len(dataset.class_names)
    model = get_model(num_classes)
    model.to(device)

    # Set up optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Train for 10 epochs

    max_epochs = 100  # Maximum number of epochs to train
    early_stopping_patience = 25  # Stop if no improvement for this many epochs
    best_loss = float("inf")
    best_epoch = 0
    no_improvement_count = 0

    # Lists to track progress
    losses = []

    print(
        f"Starting training for up to {max_epochs} epochs with early stopping patience of {early_stopping_patience}"
    )

    for epoch in range(max_epochs):
        # Train for one epoch
        loss = train_one_epoch(model, optimizer, train_loader, device)
        losses.append(loss)

        print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {loss:.4f}", end="")

        # Check if this is the best model so far
        if loss < best_loss:
            improvement = best_loss - loss
            best_loss = loss
            best_epoch = epoch
            no_improvement_count = 0

            # Save best model
            torch.save(model.state_dict(), "./models/golf_course_model_best.pth")
            print(f" ← New best model! Improvement: {improvement:.6f}")
        else:
            no_improvement_count += 1
            print(
                f" (No improvement for {no_improvement_count}/{early_stopping_patience} epochs)"
            )

        # Save regular checkpoint
        torch.save(
            model.state_dict(),
            f"./models/checkpoints/golf_course_model_epoch_{epoch + 1}.pth",
        )

        # Update learning rate
        lr_scheduler.step()

        # Early stopping check
        if no_improvement_count >= early_stopping_patience:
            print(
                f"\nEarly stopping triggered! No improvement for {early_stopping_patience} epochs."
            )
            print(f"Best loss was {best_loss:.6f} at epoch {best_epoch + 1}")
            break
    print("Training complete!")


if __name__ == "__main__":
    main()
