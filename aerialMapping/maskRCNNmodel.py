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

        self.class_stats = {
            "background": {"count": 0, "pixel_area": 0},
            "green": {"count": 0, "pixel_area": 0},
            "fairway": {"count": 0, "pixel_area": 0},
            "bunker": {"count": 0, "pixel_area": 0},
            "rough": {"count": 0, "pixel_area": 0},
            "water": {"count": 0, "pixel_area": 0},
        }

        self.sample_weights = []
        self.sample_class_counts = []

        self.parsed_data_cache = {}

        self.samples = []
        for json_file in self.json_files:
            json_path = os.path.join(json_dir, json_file)
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                img_file = data.get("imagePath", json_file.replace(".json", ".jpg"))
                img_path = os.path.join(img_dir, os.path.basename(img_file))

                if os.path.exists(img_path):
                    sample_classes = []

                    for shape in data.get("shapes", []):
                        if shape.get("shape_type") == "polygon":
                            label = shape.get("label", "unknown")
                            if label in self.class_map:
                                sample_classes.append(label)
                                self.class_stats[label]["count"] += 1

                                points = shape.get("points", [])
                                if len(points) >= 3:
                                    polygon = np.array(points, dtype=np.int32).reshape(
                                        (-1, 2)
                                    )
                                    area = cv2.contourArea(polygon)
                                    self.class_stats[label]["pixel_area"] += area

                    self.parsed_data_cache[json_path] = data

                    self.samples.append((img_path, json_path))
                    self.sample_class_counts.append(Counter(sample_classes))

                    sample_weight = 1.0
                    if "water" in sample_classes:
                        sample_weight *= 5.0
                    if "rough" in sample_classes:
                        sample_weight *= 1.5

                    self.sample_weights.append(sample_weight)

            except Exception as e:
                print(f"Error processing {json_file}: {e}")

        if self.sample_weights:
            total_weight = sum(self.sample_weights)
            self.sample_weights = [
                w / total_weight * len(self.sample_weights) for w in self.sample_weights
            ]

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
            class_counts = class_counts + 1

        weights = 1.0 / class_counts

        weights = weights / np.sum(weights) * len(weights)

        if "water" in self.class_names:
            water_idx = self.class_names.index("water")
            weights[water_idx] *= 3.0

        if "rough" in self.class_names:
            rough_idx = self.class_names.index("rough")
            weights[rough_idx] *= 1.5

        self.class_weights = torch.FloatTensor(weights)
        print(
            f"Computed class weights: {dict(zip(self.class_names, self.class_weights.numpy()))}"
        )
        return self.class_weights

    def __len__(self):
        return len(self.samples)

    def polygon_to_mask(self, polygon, img_height, img_width):
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        polygon = np.array(polygon, dtype=np.int32).reshape((-1, 2))

        cv2.fillPoly(mask, [polygon], 1)
        return mask

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        if json_path in self.parsed_data_cache:
            data = self.parsed_data_cache[json_path]
        else:
            with open(json_path, "r") as f:
                data = json.load(f)
                self.parsed_data_cache[json_path] = data

        boxes = []
        masks = []
        labels = []

        for shape in data.get("shapes", []):
            if shape.get("shape_type") == "polygon":
                points = shape.get("points", [])
                if len(points) >= 3:
                    label = shape.get("label", "unknown")
                    if label in self.class_map:
                        mask = self.polygon_to_mask(points, height, width)

                        pos = np.where(mask)
                        if len(pos[0]) > 0 and len(pos[1]) > 0:
                            xmin = np.min(pos[1])
                            xmax = np.max(pos[1])
                            ymin = np.min(pos[0])
                            ymax = np.max(pos[0])

                            if xmax > xmin and ymax > ymin:
                                boxes.append([xmin, ymin, xmax, ymax])
                                masks.append(mask)
                                labels.append(self.class_map[label])

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            masks = torch.zeros((0, height, width), dtype=torch.uint8)

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

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = torchvision.transforms.functional.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = torchvision.transforms.functional.hflip(image)

            h, w = image.shape[-2:]
            if target["boxes"].shape[0] > 0:
                target["boxes"][:, [0, 2]] = w - target["boxes"][:, [2, 0]]

            if "masks" in target:
                target["masks"] = torch.flip(target["masks"], [2])

        return image, target


class RandomRotation:
    def __init__(self, degrees=10):
        self.degrees = degrees

    def __call__(self, image, target):
        angle = random.uniform(-self.degrees, self.degrees)

        image = torchvision.transforms.functional.rotate(image, angle, expand=True)

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
        image = self.transform(image)

        return image, target


def get_transform(train):
    transforms = []

    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))

        if random.random() < 0.5:
            transforms.append(
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            )

    return Compose(transforms)


def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


def train_one_epoch(model, optimizer, data_loader, device, class_weights=None):
    model.train()
    total_loss = 0

    class_stats = {"loss": {}, "count": {}}

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        if class_weights is not None:
            if "loss_classifier" in loss_dict:
                loss_dict["loss_classifier"] *= 1.2

            if "loss_mask" in loss_dict:
                loss_dict["loss_mask"] *= 1.5

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(data_loader)


def evaluate(model, data_loader, device):
    model.eval()

    class_correct = {i: 0 for i in range(6)}
    class_total = {i: 0 for i in range(6)}

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for i, (output, target) in enumerate(zip(outputs, targets)):
                if isinstance(output, dict) and "labels" in output:
                    pred_labels = output["labels"]
                    true_labels = target["labels"]

                    for j, label in enumerate(true_labels):
                        label_idx = label.item()
                        class_total[label_idx] = class_total.get(label_idx, 0) + 1

                    for j, label in enumerate(pred_labels):
                        if j < len(true_labels) and label == true_labels[j]:
                            label_idx = label.item()
                            class_correct[label_idx] = (
                                class_correct.get(label_idx, 0) + 1
                            )

    class_accuracy = {}
    for cls_idx in class_total:
        if class_total[cls_idx] > 0:
            accuracy = class_correct[cls_idx] / class_total[cls_idx]
        else:
            accuracy = 0
        class_accuracy[cls_idx] = accuracy

    mean_accuracy = sum(class_accuracy.values()) / len(class_accuracy)

    return mean_accuracy, class_accuracy


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    num_workers = 4 if torch.cuda.is_available() else 0

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    full_dataset = GolfCourseDataset(
        img_dir="./imgs/rawImgs/",
        json_dir="./imgs/annotationsTesting/",
        transforms=None,
        compute_class_weights=True,
    )

    class_weights = getattr(full_dataset, "class_weights", None)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"Using class weights: {class_weights}")

    indices = torch.randperm(len(full_dataset)).tolist()
    train_size = int(0.8 * len(full_dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    print(f"Total samples: {len(full_dataset)}")
    print(f"Train samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")

    train_dataset_with_transforms = GolfCourseDataset(
        img_dir="./imgs/rawImgs/",
        json_dir="./imgs/annotationsTesting/",
        transforms=get_transform(train=True),
        compute_class_weights=False,
    )

    val_dataset_with_transforms = GolfCourseDataset(
        img_dir="./imgs/rawImgs/",
        json_dir="./imgs/annotationsTesting/",
        transforms=get_transform(train=False),
        compute_class_weights=False,
    )

    train_dataset = torch.utils.data.Subset(
        train_dataset_with_transforms, train_indices
    )
    val_dataset = torch.utils.data.Subset(val_dataset_with_transforms, val_indices)

    train_sample_weights = [full_dataset.sample_weights[i] for i in train_indices]
    train_sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_indices),
        replacement=True,
    )

    print("Testing dataset access...")
    try:
        train_sample = train_dataset[0]
        print(f"Train dataset working - first sample loaded")

        val_sample = val_dataset[0]
        print(f"Val dataset working - first sample loaded")

        if len(val_dataset) > 0:
            val_sample = val_dataset[len(val_dataset) - 1]
            print(f"Val dataset working - last sample loaded")

    except Exception as e:
        print(f"Dataset test failed: {e}")
        return

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

    num_classes = len(full_dataset.class_names)
    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.0001)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    max_epochs = 100
    early_stopping_patience = 20
    best_loss = float("inf")
    best_epoch = 0
    no_improvement_count = 0
    validation_frequency = 1

    train_losses = []
    val_metrics = []

    print(
        f"Starting training for up to {max_epochs} epochs with early stopping patience of {early_stopping_patience}"
    )

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, class_weights=class_weights
        )
        train_losses.append(train_loss)

        if epoch % validation_frequency == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for images, targets in val_loader:
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    model.train()
                    loss_dict = model(images, targets)
                    model.eval()

                    val_loss += sum(loss for loss in loss_dict.values())

                val_loss = val_loss / len(val_loader)

                if epoch % (validation_frequency * 2) == 0:
                    mean_accuracy, class_accuracy = evaluate(model, val_loader, device)
                else:
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

            lr_scheduler.step(val_loss)

            print(
                f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            if epoch % (validation_frequency * 2) == 0:
                print(f"Per-class accuracy: {class_accuracy}")

            if val_loss < best_loss:
                improvement = best_loss - val_loss
                best_loss = val_loss
                best_epoch = epoch
                no_improvement_count = 0

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

            if no_improvement_count >= early_stopping_patience:
                print(
                    f"\nEarly stopping triggered! No improvement for {early_stopping_patience} epochs."
                )
                print(f"Best loss was {best_loss:.6f} at epoch {best_epoch + 1}")
                break
        else:
            print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}")

    print("Training complete!")


if __name__ == "__main__":
    main()
