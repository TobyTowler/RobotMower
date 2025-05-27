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


def calculate_detection_accuracy(
    model, data_loader, device, confidence_threshold=0.5, iou_threshold=0.5
):
    """
    Calculate QUALITY-BASED detection accuracy using MASK IoU.

    Counts an object as "correctly detected" only if:
    1. Predicted class matches ground truth class
    2. Mask IoU > threshold (good segmentation quality)

    This measures: "How many objects were detected with good segmentation quality?"
    """
    model.eval()

    correct_detections = 0
    total_ground_truth = 0

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

                # Ground truth objects to find
                true_labels = target["labels"]
                true_masks = target["masks"]
                total_ground_truth += len(true_labels)

                # Match predictions to ground truth using MASK IoU
                if len(pred_masks) > 0 and len(true_masks) > 0:
                    matched_pred = set()

                    # For each ground truth object, find best matching prediction
                    for i, true_label in enumerate(true_labels):
                        true_label_idx = true_label.item()
                        best_iou = 0
                        best_pred_idx = -1

                        # Check all predictions of the same class
                        for j, pred_label in enumerate(pred_labels):
                            if j in matched_pred:  # Skip already matched predictions
                                continue
                            if pred_label.item() == true_label_idx:
                                # Calculate mask IoU (pixel-level overlap quality)
                                pred_mask = pred_masks[j].squeeze()
                                true_mask = true_masks[i].float()
                                mask_iou = calculate_mask_iou(pred_mask, true_mask)

                                if mask_iou > best_iou:
                                    best_iou = mask_iou
                                    best_pred_idx = j

                        # Count as correct detection only if IoU is high enough (quality-based)
                        if best_iou > iou_threshold and best_pred_idx != -1:
                            correct_detections += 1
                            matched_pred.add(best_pred_idx)

    detection_accuracy = (
        correct_detections / total_ground_truth if total_ground_truth > 0 else 0
    )
    return detection_accuracy


def main():
    # Confidence thresholds from 0.0 to 1.0 in 0.1 increments
    x = [i / 10.0 for i in range(10)]  # [0.0, 0.1, 0.2, ..., 1.0]
    detection_accuracies = []

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

    print("Testing QUALITY-BASED detection accuracy with confidence thresholds...")
    print("Uses MASK IoU (threshold 0.5) to determine if detection is 'good enough'")
    print("Counts objects as detected only if segmentation quality is sufficient")
    print("Higher confidence threshold = only include more confident predictions")
    print(
        "Expected: Detection accuracy should decrease with higher confidence threshold"
    )
    print("=" * 75)

    for i, threshold in enumerate(x):
        thisAccuracies = []
        for j in range(3):  # Run multiple times for averaging
            print(f"Confidence Threshold: {threshold:.1f}, Run: {j + 1}")

            detection_accuracy = calculate_detection_accuracy(
                model, test_loader, device, threshold
            )
            thisAccuracies.append(detection_accuracy)

        avg_accuracy = statistics.mean(thisAccuracies)
        detection_accuracies.append(avg_accuracy)
        print(f"  Average detection accuracy: {avg_accuracy:.4f}")

    # Plot results
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({"font.size": 22})
    plt.rcParams["savefig.directory"] = os.path.expanduser(
        "~/Programming/RobotMower/finalReport/images/"
    )

    plt.scatter(x, detection_accuracies, color="blue", marker="o", s=100, alpha=0.7)
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Detection Accuracy")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
