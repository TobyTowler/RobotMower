import json
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def polygon_to_mask(polygon, img_height, img_width):
    """Convert polygon to binary mask"""
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    # Convert polygon to numpy array of shape (N, 2)
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 2))
    # Fill polygon
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def parse_labelme_json(json_path, img_path=None):
    """Parse LabelMe JSON file and return image and masks"""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Get image dimensions
    if img_path is None:
        img_path = data["imagePath"]
    img = np.array(Image.open(os.path.join(os.path.dirname(json_path), img_path)))
    img_height, img_width = img.shape[:2]

    # Create masks for each shape
    masks = []
    class_names = []

    for shape in data["shapes"]:
        if shape["shape_type"] == "polygon":
            points = shape["points"]
            label = shape["label"]
            mask = polygon_to_mask(points, img_height, img_width)
            masks.append(mask)
            class_names.append(label)

    return img, masks, class_names


class PolygonDataset(Dataset):
    def __init__(self, img_dir, json_dir, transform=None):
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.transform = transform
        self.json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

        # Create a mapping of class names to indices
        self.class_map = {}
        self.samples = []

        for json_file in self.json_files:
            json_path = os.path.join(json_dir, json_file)

            try:
                # Check if JSON file is valid
                with open(json_path, "r") as f:
                    data = json.load(f)

                # Get corresponding image file
                img_file = data.get("imagePath", json_file.replace(".json", ".jpg"))
                img_path = os.path.join(img_dir, os.path.basename(img_file))

                # Skip if image doesn't exist
                if not os.path.exists(img_path):
                    print(f"Warning: Image {img_path} not found, skipping")
                    continue

                # Check if there are polygon annotations
                has_polygons = False
                for shape in data.get("shapes", []):
                    if shape.get("shape_type") == "polygon":
                        has_polygons = True
                        class_name = shape.get("label", "unknown")
                        if class_name not in self.class_map:
                            self.class_map[class_name] = len(self.class_map)

                # Add to samples regardless of whether it has polygons
                # This allows us to handle images with no annotations
                self.samples.append((img_path, json_path, has_polygons))

            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                print(f"Error processing {json_file}: {e}")
                continue

        print(f"Loaded {len(self.samples)} samples with {len(self.class_map)} classes")
        print(f"Class mapping: {self.class_map}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path, has_polygons = self.samples[idx]

        # Load image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy black image and empty targets
            img = Image.new("RGB", (224, 224), color=0)
            return (
                self.transform(img) if self.transform else transforms.ToTensor()(img),
                torch.zeros((1, 224, 224)),
                torch.zeros(len(self.class_map)),
            )

        # Original image dimensions for mask creation
        width, height = img.size

        # Apply transformations to image
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        # Return early if no polygons
        if not has_polygons:
            return (
                img_tensor,
                torch.zeros((1, height, width)).float(),
                torch.zeros(len(self.class_map)),
            )

        # Try to load and process annotations
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            masks = []
            class_names = []

            # Process each shape
            for shape in data.get("shapes", []):
                if shape.get("shape_type") == "polygon":
                    points = shape.get("points", [])
                    if len(points) >= 3:  # Need at least 3 points for a polygon
                        label = shape.get("label", "unknown")
                        mask = polygon_to_mask(points, height, width)
                        masks.append(mask)
                        class_names.append(label)

            # Create target tensors
            if len(masks) > 0:
                # For segmentation: stack all masks
                mask = np.stack(masks, axis=0)
                seg_target = torch.from_numpy(mask).float()

                # For classification: create one-hot encoded target
                class_indices = [self.class_map.get(name, 0) for name in class_names]
                class_target = torch.zeros(len(self.class_map))
                for idx in class_indices:
                    class_target[idx] = 1
            else:
                # No valid polygons found
                seg_target = torch.zeros((1, height, width)).float()
                class_target = torch.zeros(len(self.class_map))

        except Exception as e:
            print(f"Error processing annotations for {json_path}: {e}")
            seg_target = torch.zeros((1, height, width)).float()
            class_target = torch.zeros(len(self.class_map))

        return img_tensor, seg_target, class_target
