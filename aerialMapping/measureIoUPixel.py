import os
import matplotlib.pyplot as plt
import statistics
import torch
import numpy as np
from torch.utils.data import DataLoader
from testWeights import load_model
from maskRCNNmodel import GolfCourseDataset, get_transform


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

    print("Testing pixel accuracy with mask IoU thresholds...")
    print("Only measuring accuracy on NON-BACKGROUND pixels (golf course features)")
    print("Higher IoU threshold = only include more precise predictions")
    print("Expected: Higher IoU threshold should give higher pixel accuracy")
    print("=" * 60)

    for i, threshold in enumerate(x):
        thisAccuracies = []
        for j in range(3):
            print(f"Mask IoU Threshold: {threshold:.1f}, Run: {j + 1}")

            pixel_accuracy = calculate_pixel_accuracy_with_mask_iou(
                model, test_loader, device, threshold
            )
            thisAccuracies.append(pixel_accuracy)

        avg_accuracy = statistics.mean(thisAccuracies)
        pixel_accuracies.append(avg_accuracy)
        print(f"  Average pixel accuracy: {avg_accuracy:.4f}")

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    for threshold, accuracy in zip(x, pixel_accuracies):
        print(f"IoU {threshold:.1f}: {accuracy:.4f}")

    plt.figure(figsize=(10, 10))
    plt.rcParams.update({"font.size": 16})

    plt.scatter(x, pixel_accuracies, color="blue", marker="o", s=100, alpha=0.7)
    plt.xlabel("Mask IoU Threshold")
    plt.ylabel("Pixel Accuracy")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    if len(pixel_accuracies) > 1:
        trend = (
            "increasing" if pixel_accuracies[-1] > pixel_accuracies[0] else "decreasing"
        )
        print(f"\nOverall trend: {trend}")
        if trend == "decreasing":
            print("This suggests many predictions have low mask IoU overlap")
        else:
            print(
                "This suggests good mask precision - higher IoU gives better pixel accuracy"
            )


if __name__ == "__main__":
    main()
