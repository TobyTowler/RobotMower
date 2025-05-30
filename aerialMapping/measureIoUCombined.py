import os
import matplotlib.pyplot as plt
import statistics
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from testWeights import load_model, visualize_prediction
from maskRCNNmodel import GolfCourseDataset, get_transform


def calculate_accuracy_with_iou_threshold(
    model, data_loader, device, iou_threshold=0.5, conf_threshold=0.5
):
    model.eval()

    correct_detections = 0
    total_ground_truth = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)

            for img_tensor, pred, target in zip(images, predictions, targets):
                if len(pred["scores"]) > 0:
                    high_conf_indices = pred["scores"] > conf_threshold
                    pred_boxes = pred["boxes"][high_conf_indices]
                    pred_labels = pred["labels"][high_conf_indices]
                else:
                    pred_boxes = torch.empty((0, 4))
                    pred_labels = torch.empty(0, dtype=torch.long)

                true_boxes = target["boxes"]
                true_labels = target["labels"]
                total_ground_truth += len(true_labels)

                if len(pred_boxes) > 0 and len(true_boxes) > 0:
                    box_ious = box_iou(pred_boxes, true_boxes)

                    matched_pred = set()

                    for i, true_label in enumerate(true_labels):
                        true_label_idx = true_label.item()
                        best_iou = 0
                        best_pred_idx = -1

                        for j, pred_label in enumerate(pred_labels):
                            if j in matched_pred:
                                continue
                            if (
                                pred_label.item() == true_label_idx
                                and box_ious[j, i] > best_iou
                            ):
                                best_iou = box_ious[j, i].item()
                                best_pred_idx = j

                        if best_iou > iou_threshold and best_pred_idx != -1:
                            correct_detections += 1
                            matched_pred.add(best_pred_idx)

    accuracy = correct_detections / total_ground_truth if total_ground_truth > 0 else 0
    return accuracy


def calculate_mask_iou(pred_mask, true_mask):
    if pred_mask.sum() == 0 and true_mask.sum() == 0:
        return 1.0

    pred_binary = (pred_mask > 0.5).float()
    true_binary = true_mask.float()

    intersection = (pred_binary * true_binary).sum()
    union = pred_binary.sum() + true_binary.sum() - intersection

    if union == 0:
        return 0.0

    return (intersection / union).item()


def calculate_pixel_accuracy_with_mask_iou(
    model, data_loader, device, iou_threshold=0.5, conf_threshold=0.5
):
    model.eval()

    total_pixel_correct = 0
    total_measured_pixels = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)

            for img_tensor, pred, target in zip(images, predictions, targets):
                img_height, img_width = img_tensor.shape[-2:]

                if len(pred["scores"]) > 0:
                    high_conf_indices = pred["scores"] > conf_threshold
                    pred_labels = pred["labels"][high_conf_indices]
                    pred_masks = pred["masks"][high_conf_indices]
                else:
                    pred_labels = torch.empty(0, dtype=torch.long)
                    pred_masks = torch.empty((0, img_height, img_width))

                true_labels = target["labels"]
                true_masks = target["masks"]

                gt_pixel_map = torch.zeros(
                    (img_height, img_width), device=device, dtype=torch.long
                )
                for i, mask in enumerate(true_masks):
                    label_val = true_labels[i].item()
                    gt_pixel_map[mask.bool()] = label_val

                if len(pred_masks) > 0 and len(true_masks) > 0:
                    for j, pred_label in enumerate(pred_labels):
                        pred_label_idx = pred_label.item()
                        pred_mask = pred_masks[j].squeeze()

                        best_iou = 0
                        for i, true_label in enumerate(true_labels):
                            true_label_idx = true_label.item()

                            if pred_label_idx == true_label_idx:
                                true_mask = true_masks[i].float()
                                mask_iou = calculate_mask_iou(pred_mask, true_mask)
                                if mask_iou > best_iou:
                                    best_iou = mask_iou

                        if best_iou >= iou_threshold:
                            pred_binary = pred_mask > 0.5
                            if pred_binary.sum() > 0:
                                gt_pixels = gt_pixel_map[pred_binary]
                                pred_pixels = torch.full_like(gt_pixels, pred_label_idx)

                                correct = (gt_pixels == pred_pixels).sum().item()
                                total_pixel_correct += correct
                                total_measured_pixels += len(gt_pixels)

    pixel_accuracy = (
        total_pixel_correct / total_measured_pixels if total_measured_pixels > 0 else 0
    )
    return pixel_accuracy


def main():
    x = [i / 10.0 for i in range(1, 10)]
    detection_accuracies = []
    pixel_accuracies = []

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    model_path = "./models/golf_course_model_best.pth"
    model = load_model(model_path, 6)
    model.to(device)

    test_dataset = GolfCourseDataset(
        img_dir="./imgs/rawImgs/",
        json_dir="./imgs/annotationsTesting/",
        transforms=get_transform(train=False),
        compute_class_weights=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    print("Testing both detection and pixel accuracy with IoU thresholds...")
    print("=" * 70)

    for i, threshold in enumerate(x):
        print(f"\nTesting IoU threshold: {threshold:.1f}")

        detection_runs = []
        for j in range(3):
            detection_acc = calculate_accuracy_with_iou_threshold(
                model, test_loader, device, threshold
            )
            detection_runs.append(detection_acc)

        avg_detection_acc = statistics.mean(detection_runs)
        detection_accuracies.append(avg_detection_acc)

        pixel_runs = []
        for j in range(3):
            pixel_acc = calculate_pixel_accuracy_with_mask_iou(
                model, test_loader, device, threshold
            )
            pixel_runs.append(pixel_acc)

        avg_pixel_acc = statistics.mean(pixel_runs)
        pixel_accuracies.append(avg_pixel_acc)

        print(f"  Detection Accuracy: {avg_detection_acc:.4f}")
        print(f"  Pixel Accuracy: {avg_pixel_acc:.4f}")

    plt.figure(figsize=(12, 8))
    plt.rcParams.update({"font.size": 14})

    plt.plot(
        x,
        detection_accuracies,
        "o-",
        color="blue",
        linewidth=2,
        markersize=8,
        label="Detection Accuracy (Box IoU)",
        alpha=0.8,
    )
    plt.plot(
        x,
        pixel_accuracies,
        "s-",
        color="red",
        linewidth=2,
        markersize=8,
        label="Pixel Accuracy (Mask IoU)",
        alpha=0.8,
    )

    plt.xlabel("IoU Threshold", fontsize=16)
    plt.ylabel("Accuracy %", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=14, loc="best")

    plt.xlim(0.05, 0.95)
    plt.ylim(0, max(max(detection_accuracies), max(pixel_accuracies)) * 1.1)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
