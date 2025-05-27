import os
import matplotlib.pyplot as plt
import statistics
import torch
import numpy as np
from torch.utils.data import DataLoader
from testWeights import load_model
from maskRCNNmodel import GolfCourseDataset, get_transform


def calculate_mask_iou(pred_mask, true_mask):
    """Calculate IoU between predicted and true masks"""
    if pred_mask.sum() == 0 and true_mask.sum() == 0:
        return 1.0

    # Convert to binary masks
    pred_binary = (pred_mask > 0.5).float()
    true_binary = true_mask.float()

    # Calculate intersection and union
    intersection = (pred_binary * true_binary).sum()
    union = pred_binary.sum() + true_binary.sum() - intersection

    if union == 0:
        return 0.0

    return (intersection / union).item()


def calculate_pixel_accuracy_with_mask_iou(
    model, data_loader, device, confidence_threshold=0.5, iou_threshold=0.5
):
    """
    Calculate pixel accuracy using mask IoU filtering.

    Only includes predictions that have good mask IoU with ground truth,
    then measures pixel-level classification accuracy.
    """
    model.eval()

    total_pixel_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get predictions
            predictions = model(images)

            for pred, target in zip(predictions, targets):
                img_height, img_width = images[0].shape[-2:]

                # Filter predictions by confidence threshold
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

                # Create ground truth pixel map
                combined_true_mask = torch.zeros(
                    (img_height, img_width), device=device, dtype=torch.long
                )
                for i, mask in enumerate(true_masks):
                    label_val = true_labels[i].item()
                    combined_true_mask[mask.bool()] = label_val

                # Create prediction pixel map using only high-quality predictions (mask IoU filter)
                combined_pred_mask = torch.zeros(
                    (img_height, img_width), device=device, dtype=torch.long
                )

                if len(pred_masks) > 0 and len(true_masks) > 0:
                    # For each prediction, check if it has good mask IoU with any ground truth
                    for j, pred_label in enumerate(pred_labels):
                        pred_label_idx = pred_label.item()
                        pred_mask = pred_masks[j].squeeze()

                        # Find best matching ground truth of same class
                        best_iou = 0
                        for i, true_label in enumerate(true_labels):
                            true_label_idx = true_label.item()

                            if pred_label_idx == true_label_idx:
                                true_mask = true_masks[i].float()
                                mask_iou = calculate_mask_iou(pred_mask, true_mask)
                                if mask_iou > best_iou:
                                    best_iou = mask_iou

                        # Only include prediction in pixel map if IoU is good enough
                        if best_iou >= iou_threshold:
                            mask_binary = pred_mask > 0.5
                            combined_pred_mask[mask_binary] = pred_label_idx

                # Calculate pixel accuracy
                correct_pixels = (combined_true_mask == combined_pred_mask).sum().item()
                total_pixel_correct += correct_pixels
                total_pixels += img_height * img_width

    pixel_accuracy = total_pixel_correct / total_pixels if total_pixels > 0 else 0
    return pixel_accuracy


def main():
    # Confidence thresholds from 0.0 to 1.0 in 0.1 increments
    x = [i / 10.0 for i in range(10)]  # [0.0, 0.1, 0.2, ..., 1.0]
    pixel_accuracies = []

    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    model_path = "./models/golf_course_model_best.pth"
    model = load_model(model_path, 6)
    model.to(device)

    # Create test dataset
    test_dataset = GolfCourseDataset(
        img_dir="./imgs/rawImgs/",
        json_dir="./imgs/annotationsTesting/",
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

    print("Testing PIXEL ACCURACY with confidence thresholds...")
    print("Uses MASK IoU filtering (threshold 0.5) to include only quality predictions")
    print("Then measures pixel-level classification accuracy across entire image")
    print("Higher confidence threshold = fewer predictions included")
    print("=" * 75)

    for i, threshold in enumerate(x):
        thisAccuracies = []
        for j in range(3):  # Run multiple times for averaging
            print(f"Confidence Threshold: {threshold:.1f}, Run: {j + 1}")

            pixel_accuracy = calculate_pixel_accuracy_with_mask_iou(
                model, test_loader, device, threshold
            )
            thisAccuracies.append(pixel_accuracy)

        avg_accuracy = statistics.mean(thisAccuracies)
        pixel_accuracies.append(avg_accuracy)
        print(f"  Average pixel accuracy: {avg_accuracy:.4f}")

    # Plot results
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({"font.size": 16})
    plt.rcParams["savefig.directory"] = os.path.expanduser(
        "~/Programming/RobotMower/finalReport/images/"
    )

    plt.scatter(x, pixel_accuracies, color="blue", marker="o", s=100, alpha=0.7)
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Pixel Accuracy")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
