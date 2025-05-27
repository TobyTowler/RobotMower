import os
import matplotlib.pyplot as plt
import statistics
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from testWeights import load_model
from maskRCNNmodel import GolfCourseDataset, get_transform


class ResizeDataset(GolfCourseDataset):
    """Dataset that resizes images to specific resolution"""

    def __init__(
        self,
        img_dir,
        json_dir,
        target_size,
        transforms=None,
        compute_class_weights=False,
    ):
        super().__init__(img_dir, json_dir, transforms, compute_class_weights)
        self.target_size = target_size
        self.resize_transform = transforms.Resize((target_size, target_size))

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]

        # Load and resize image
        img = Image.open(img_path).convert("RGB")
        original_width, original_height = img.size
        img = self.resize_transform(img)

        # Calculate scaling factors
        scale_x = self.target_size / original_width
        scale_y = self.target_size / original_height

        # Load annotations from cache
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

        # Process each shape with scaling
        for shape in data.get("shapes", []):
            if shape.get("shape_type") == "polygon":
                points = shape.get("points", [])
                if len(points) >= 3:
                    label = shape.get("label", "unknown")
                    if label in self.class_map:
                        # Scale points to new image size
                        scaled_points = [
                            [p[0] * scale_x, p[1] * scale_y] for p in points
                        ]

                        # Create mask
                        mask = self.polygon_to_mask(
                            scaled_points, self.target_size, self.target_size
                        )

                        # Find bounding box
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
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            masks = torch.zeros(
                (0, self.target_size, self.target_size), dtype=torch.uint8
            )

        # Create target
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

        # Apply additional transforms if provided
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def calculate_pixel_accuracy_resolution(
    model, data_loader, device, confidence_threshold=0.5
):
    """Calculate pixel accuracy for specific resolution"""
    model.eval()

    total_pixel_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)

            for pred, target in zip(predictions, targets):
                img_height, img_width = images[0].shape[-2:]

                # Filter predictions by confidence
                if len(pred["scores"]) > 0:
                    high_conf_indices = pred["scores"] > confidence_threshold
                    pred_labels = pred["labels"][high_conf_indices]
                    pred_masks = pred["masks"][high_conf_indices]
                else:
                    pred_labels = torch.empty(0, dtype=torch.long)
                    pred_masks = torch.empty((0, img_height, img_width))

                # Ground truth
                true_labels = target["labels"]
                true_masks = target["masks"]

                # Create combined masks
                combined_true_mask = torch.zeros((img_height, img_width), device=device)
                for i, mask in enumerate(true_masks):
                    label_val = true_labels[i].item()
                    combined_true_mask[mask.bool()] = label_val

                combined_pred_mask = torch.zeros((img_height, img_width), device=device)
                if len(pred_masks) > 0:
                    for i, mask in enumerate(pred_masks):
                        label_val = pred_labels[i].item()
                        mask_binary = mask.squeeze() > 0.5
                        combined_pred_mask[mask_binary] = label_val

                # Calculate pixel accuracy
                correct_pixels = (combined_true_mask == combined_pred_mask).sum().item()
                total_pixel_correct += correct_pixels
                total_pixels += img_height * img_width

    pixel_accuracy = total_pixel_correct / total_pixels if total_pixels > 0 else 0
    return pixel_accuracy


def main():
    # Image resolutions to test
    x = [256, 384, 512, 640, 768, 896, 1024]
    accuracies = []

    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    model_path = "./models/golf_course_model_best.pth"
    model = load_model(model_path, 6)
    model.to(device)

    for i, resolution in enumerate(x):
        thisAccuracies = []
        for j in range(3):  # Run multiple times for averaging
            print(f"Resolution: {resolution}x{resolution}, Run: {j + 1}")

            # Create dataset with specific resolution
            test_dataset = ResizeDataset(
                img_dir="./imgs/rawImgs/",
                json_dir="./imgs/annotationsTesting/",
                target_size=resolution,
                transforms=get_transform(train=False),
                compute_class_weights=False,
            )

            # Create test loader
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=lambda x: tuple(zip(*x)),
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False,
            )

            accuracy = calculate_pixel_accuracy_resolution(model, test_loader, device)
            thisAccuracies.append(accuracy)

        accuracies.append(statistics.mean(thisAccuracies))

    # Plot results
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({"font.size": 16})
    plt.rcParams["savefig.directory"] = os.path.expanduser(
        "~/Programming/RobotMower/finalReport/images/"
    )

    plt.scatter(x, accuracies, color="blue", marker="o", s=100, alpha=0.7)
    plt.xlabel("Image Resolution (pixels)")
    plt.ylabel("Pixel Accuracy")
    plt.title("Model Pixel Accuracy vs Image Resolution")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
