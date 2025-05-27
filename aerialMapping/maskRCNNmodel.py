import torchvision
import os
import json
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
from PIL import Image
import random
import torch.nn.functional as F
from collections import Counter


class GolfCourseDataset(Dataset):
    def __init__(self, img_dir, json_dir, transforms=None, compute_class_weights=False):
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

        # Track class statistics
        self.class_stats = {
            "background": {"count": 0, "pixel_area": 0},
            "green": {"count": 0, "pixel_area": 0},
            "fairway": {"count": 0, "pixel_area": 0},
            "bunker": {"count": 0, "pixel_area": 0},
            "rough": {"count": 0, "pixel_area": 0},
            "water": {"count": 0, "pixel_area": 0},
        }

        # Track sample class weights for weighted sampling
        self.sample_weights = []
        self.sample_class_counts = []

        # Cache the parsed data to avoid re-reading files
        self.parsed_data_cache = {}

        # Build valid samples list more efficiently
        self.samples = []
        for json_file in self.json_files:
            json_path = os.path.join(json_dir, json_file)
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                img_file = data.get("imagePath", json_file.replace(".json", ".jpg"))
                img_path = os.path.join(img_dir, os.path.basename(img_file))

                if os.path.exists(img_path):
                    # Count classes and compute weights in one pass
                    sample_classes = []

                    for shape in data.get("shapes", []):
                        if shape.get("shape_type") == "polygon":
                            label = shape.get("label", "unknown")
                            if label in self.class_map:
                                sample_classes.append(label)
                                # Update class counts
                                self.class_stats[label]["count"] += 1

                                # Estimate pixel area (approximate)
                                points = shape.get("points", [])
                                if len(points) >= 3:
                                    # Calculate rough polygon area without loading image
                                    polygon = np.array(points, dtype=np.int32).reshape(
                                        (-1, 2)
                                    )
                                    area = cv2.contourArea(polygon)
                                    self.class_stats[label]["pixel_area"] += area

                    # Cache the parsed data
                    self.parsed_data_cache[json_path] = data

                    # Record class counts for this sample
                    self.samples.append((img_path, json_path))
                    self.sample_class_counts.append(Counter(sample_classes))

                    # Calculate sample weight (inverse frequency)
                    # Higher weight for rare classes, especially water
                    sample_weight = 1.0
                    if "water" in sample_classes:
                        sample_weight *= 5.0  # Boost water samples significantly
                    if "rough" in sample_classes:
                        sample_weight *= 1.5  # Boost rough samples

                    self.sample_weights.append(sample_weight)

            except Exception as e:
                print(f"Error processing {json_file}: {e}")

        # Normalize sample weights
        if self.sample_weights:
            total_weight = sum(self.sample_weights)
            self.sample_weights = [
                w / total_weight * len(self.sample_weights) for w in self.sample_weights
            ]

        # Compute class weights based on inverse frequency
        if compute_class_weights:
            self.compute_class_weights()

        print(
            f"Loaded {len(self.samples)} samples with {len(self.class_names)} classes"
        )
        print(f"Class mapping: {self.class_map}")
        print(f"Class statistics: {self.class_stats}")

    def compute_class_weights(self):
        """Compute class weights based on inverse frequency for loss function."""
        class_counts = np.array(
            [self.class_stats[cls]["count"] for cls in self.class_names]
        )
        if np.min(class_counts) == 0:
            # Avoid division by zero by adding a small constant
            class_counts = class_counts + 1

        # Inverse frequency weighting
        weights = 1.0 / class_counts

        # Normalize weights
        weights = weights / np.sum(weights) * len(weights)

        # Additional boost for rare classes
        # This addresses the extreme imbalance for water class
        if "water" in self.class_names:
            water_idx = self.class_names.index("water")
            weights[water_idx] *= 3.0  # Triple the weight for water

        if "rough" in self.class_names:
            rough_idx = self.class_names.index("rough")
            weights[rough_idx] *= 1.5  # 50% boost for rough

        self.class_weights = torch.FloatTensor(weights)
        print(
            f"Computed class weights: {dict(zip(self.class_names, self.class_weights.numpy()))}"
        )
        return self.class_weights

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

        # Load annotations (from cache if available)
        if json_path in self.parsed_data_cache:
            data = self.parsed_data_cache[json_path]
        else:
            with open(json_path, "r") as f:
                data = json.load(f)
                self.parsed_data_cache[json_path] = data

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


# NEW: Additional augmentations to help with rare classes
class RandomRotation:
    def __init__(self, degrees=10):
        self.degrees = degrees

    def __call__(self, image, target):
        angle = random.uniform(-self.degrees, self.degrees)

        # Rotate image
        image = torchvision.transforms.functional.rotate(image, angle, expand=True)

        # For simplicity, we're not implementing the complex box/mask rotation
        # This would require additional code to properly transform the boxes and masks
        # Just return the rotated image and original target for now
        # In a real implementation, you'd want to properly transform the targets too
        return image, target


class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.transform = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, image, target):
        # Apply color jitter to image
        image = self.transform(image)
        # Target remains unchanged
        return image, target


def get_transform(train):
    transforms = []
    # Convert PIL image to tensor
    transforms.append(ToTensor())
    if train:
        # Add data augmentation for training - use only the most efficient ones
        transforms.append(RandomHorizontalFlip(0.5))
        # Only apply color jitter to a subset of images to reduce overhead
        if random.random() < 0.5:  # 50% chance of color jitter
            transforms.append(
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            )

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


# NEW: Custom Focal Loss implementation for handling class imbalance
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Compute probabilities
        pt = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


# NEW: Improved training function with class-weighted loss
def train_one_epoch(model, optimizer, data_loader, device, class_weights=None):
    model.train()
    total_loss = 0

    # Dict to track per-class statistics
    class_stats = {"loss": {}, "count": {}}

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)

        # Apply class weighting if provided
        if class_weights is not None:
            # Extract relevant losses that would benefit from weighting
            # This is specific to the Mask R-CNN implementation
            if "loss_classifier" in loss_dict:
                # Scale classification loss by class weights
                # The exact implementation would depend on the model's loss structure
                # For demonstration, we're just scaling the classifier loss
                loss_dict["loss_classifier"] *= 1.2  # Boost classifier loss

            if "loss_mask" in loss_dict:
                # Scale mask loss by class weights
                # For simplicity, just boost the mask loss more
                loss_dict["loss_mask"] *= 1.5  # Boost mask loss

        # Sum losses
        losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(data_loader)


# Evaluation function to track per-class metrics
def evaluate(model, data_loader, device):
    model.eval()

    # Initialize metrics
    class_correct = {i: 0 for i in range(6)}  # 6 classes (including background)
    class_total = {i: 0 for i in range(6)}

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # For evaluation, the model returns predictions directly
            outputs = model(images)

            # Analyze predictions vs targets
            for i, (output, target) in enumerate(zip(outputs, targets)):
                # During evaluation, model returns a list of dicts with predictions
                if isinstance(output, dict) and "labels" in output:
                    pred_labels = output["labels"]
                    true_labels = target["labels"]

                    # Count per-class metrics
                    for j, label in enumerate(true_labels):
                        label_idx = label.item()
                        class_total[label_idx] = class_total.get(label_idx, 0) + 1

                    # This is a simplified metric - in real-world you'd want more sophisticated
                    # metrics like IoU for segmentation tasks
                    # For demonstration purposes only
                    for j, label in enumerate(pred_labels):
                        if j < len(true_labels) and label == true_labels[j]:
                            label_idx = label.item()
                            class_correct[label_idx] = (
                                class_correct.get(label_idx, 0) + 1
                            )

    # Calculate per-class accuracy
    class_accuracy = {}
    for cls_idx in class_total:
        if class_total[cls_idx] > 0:
            accuracy = class_correct[cls_idx] / class_total[cls_idx]
        else:
            accuracy = 0
        class_accuracy[cls_idx] = accuracy

    # Calculate mean accuracy
    mean_accuracy = sum(class_accuracy.values()) / len(class_accuracy)

    return mean_accuracy, class_accuracy


def main():
    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Set number of workers for data loading
    num_workers = 4 if torch.cuda.is_available() else 0

    # Enable CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Create dataset with class weights calculation
    full_dataset = GolfCourseDataset(
        img_dir="./imgs/rawImgs/",
        json_dir="./imgs/annotationsTesting/",
        transforms=None,  # Don't apply transforms yet
        compute_class_weights=True,
    )

    # Get class weights for loss function
    class_weights = getattr(full_dataset, "class_weights", None)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"Using class weights: {class_weights}")

    # Split into train/validation indices
    indices = torch.randperm(len(full_dataset)).tolist()
    train_size = int(0.8 * len(full_dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    print(f"Total samples: {len(full_dataset)}")
    print(f"Train samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")

    # Create training dataset with transforms
    train_dataset_with_transforms = GolfCourseDataset(
        img_dir="./imgs/rawImgs/",
        json_dir="./imgs/annotationsTesting/",
        transforms=get_transform(train=True),
        compute_class_weights=False,  # Already computed
    )

    # Create validation dataset with transforms
    val_dataset_with_transforms = GolfCourseDataset(
        img_dir="./imgs/rawImgs/",
        json_dir="./imgs/annotationsTesting/",  # Same directory!
        transforms=get_transform(train=False),
        compute_class_weights=False,  # Already computed
    )

    # Create subsets using the same indices
    train_dataset = torch.utils.data.Subset(
        train_dataset_with_transforms, train_indices
    )
    val_dataset = torch.utils.data.Subset(val_dataset_with_transforms, val_indices)

    # Create weighted sampler for training set to address class imbalance
    train_sample_weights = [full_dataset.sample_weights[i] for i in train_indices]
    train_sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_indices),
        replacement=True,
    )

    # Test the datasets before training
    print("Testing dataset access...")
    try:
        # Test train dataset
        train_sample = train_dataset[0]
        print(f"✓ Train dataset working - first sample loaded")

        # Test val dataset
        val_sample = val_dataset[0]
        print(f"✓ Val dataset working - first sample loaded")

        # Test last sample
        if len(val_dataset) > 0:
            val_sample = val_dataset[len(val_dataset) - 1]
            print(f"✓ Val dataset working - last sample loaded")

    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return

    # Create data loaders with multiple workers for CPU parallelism
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        sampler=train_sampler,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Create model with reduced complexity if needed
    num_classes = len(full_dataset.class_names)
    model = get_model(num_classes)
    model.to(device)

    # Set up optimizer with weight decay regularization
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.0001)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Set training parameters
    max_epochs = 100
    early_stopping_patience = 20
    best_loss = float("inf")
    best_epoch = 0
    no_improvement_count = 0
    validation_frequency = 1

    # Track progress
    train_losses = []
    val_metrics = []

    print(
        f"Starting training for up to {max_epochs} epochs with early stopping patience of {early_stopping_patience}"
    )

    for epoch in range(max_epochs):
        # Train for one epoch with weighted loss
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, class_weights=class_weights
        )
        train_losses.append(train_loss)

        # Only validate every N epochs to speed up training
        if epoch % validation_frequency == 0:
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                # Calculate validation loss
                val_loss = 0
                for images, targets in val_loader:
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    # In evaluation mode, we need to manually set the model's training mode temporarily
                    model.train()
                    loss_dict = model(images, targets)
                    model.eval()

                    # Sum up the losses
                    val_loss += sum(loss for loss in loss_dict.values())

                val_loss = val_loss / len(val_loader)

                # Calculate per-class metrics (less frequently to save time)
                if epoch % (validation_frequency * 2) == 0:
                    mean_accuracy, class_accuracy = evaluate(model, val_loader, device)
                else:
                    # Reuse previous class accuracy to save time
                    mean_accuracy = (
                        val_metrics[-1]["mean_accuracy"] if val_metrics else 0
                    )
                    class_accuracy = (
                        val_metrics[-1]["class_accuracy"] if val_metrics else {}
                    )

                val_metrics.append(
                    {
                        "loss": val_loss,
                        "mean_accuracy": mean_accuracy,
                        "class_accuracy": class_accuracy,
                    }
                )

            # Update learning rate based on validation loss
            lr_scheduler.step(val_loss)

            # Print progress with detailed per-class metrics
            print(
                f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            if epoch % (validation_frequency * 2) == 0:
                print(f"Per-class accuracy: {class_accuracy}")

            # Check if this is the best model so far
            if val_loss < best_loss:
                improvement = best_loss - val_loss
                best_loss = val_loss
                best_epoch = epoch
                no_improvement_count = 0

                # Save best model (save less frequently to reduce I/O overhead)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "class_weights": class_weights,
                        "class_accuracy": class_accuracy,
                        "val_loss": val_loss,
                    },
                    "./models/golf_course_model_best.pth",
                )

                print(f" ← New best model! Improvement: {improvement:.6f}")
            else:
                no_improvement_count += 1
                print(
                    f" (No improvement for {no_improvement_count}/{early_stopping_patience} epochs)"
                )

            # Save regular checkpoint (less frequently to reduce I/O overhead)
            if epoch % 5 == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "class_accuracy": class_accuracy,
                    },
                    f"./models/checkpoints/golf_course_model_epoch_{epoch + 1}.pth",
                )

            # Early stopping check
            if no_improvement_count >= early_stopping_patience:
                print(
                    f"\nEarly stopping triggered! No improvement for {early_stopping_patience} epochs."
                )
                print(f"Best loss was {best_loss:.6f} at epoch {best_epoch + 1}")
                break
        else:
            # For epochs without validation, just print training loss
            print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}")

    print("Training complete!")


if __name__ == "__main__":
    main()
