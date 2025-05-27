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
    model, data_loader, device, iou_threshold=0.5, conf_threshold=0.5, show_images=False
):
    """Calculate accuracy using different IoU thresholds for matching predictions to ground truth"""
    model.eval()

    correct_detections = 0
    total_ground_truth = 0
    batch_count = 0
    class_names = ["background", "green", "fairway", "bunker", "rough", "water"]

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get predictions
            predictions = model(images)

            for img_tensor, pred, target in zip(images, predictions, targets):
                batch_count += 1

                # Show image predictions for first few samples if requested
                if show_images and batch_count <= 5:
                    print(f"\nShowing prediction for sample {batch_count}")

                    # Convert tensor back to numpy for visualization
                    image_np = img_tensor.cpu().permute(1, 2, 0).numpy()

                    # Create result dictionary for visualization (similar to testWeights)
                    result = {
                        "image": image_np,
                        "boxes": [],
                        "masks": [],
                        "classes": [],
                        "scores": [],
                    }

                    # Filter predictions by confidence
                    if len(pred["scores"]) > 0:
                        high_conf_indices = pred["scores"] > conf_threshold

                        for box, mask, label, score in zip(
                            pred["boxes"][high_conf_indices],
                            pred["masks"][high_conf_indices],
                            pred["labels"][high_conf_indices],
                            pred["scores"][high_conf_indices],
                        ):
                            result["boxes"].append(box.cpu().numpy())
                            result["masks"].append(mask[0].cpu().numpy() > 0.5)
                            result["classes"].append(class_names[label.cpu().item()])
                            result["scores"].append(score.cpu().numpy())

                    # Visualize the prediction
                    try:
                        visualize_prediction(result, output_path=None, show=True)
                    except Exception as e:
                        print(f"Error visualizing image {batch_count}: {e}")
                        # Fallback: just show basic info
                        print(f"  Predictions: {len(result['boxes'])} objects detected")
                        print(f"  Classes: {result['classes']}")
                        print(f"  Scores: {[f'{s:.3f}' for s in result['scores']]}")

                # Continue with accuracy calculation
                if len(pred["scores"]) > 0:
                    high_conf_indices = pred["scores"] > conf_threshold
                    pred_boxes = pred["boxes"][high_conf_indices]
                    pred_labels = pred["labels"][high_conf_indices]
                else:
                    pred_boxes = torch.empty((0, 4))
                    pred_labels = torch.empty(0, dtype=torch.long)

                # Ground truth
                true_boxes = target["boxes"]
                true_labels = target["labels"]
                total_ground_truth += len(true_labels)

                # Calculate detection matches using box IoU
                if len(pred_boxes) > 0 and len(true_boxes) > 0:
                    box_ious = box_iou(pred_boxes, true_boxes)

                    # Match predictions to ground truth using IoU threshold
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


def print_detailed_results(thresholds, accuracies):
    """Print detailed results in a nice format"""
    print("\n" + "=" * 50)
    print("DETAILED IoU THRESHOLD RESULTS")
    print("=" * 50)
    print(f"{'IoU Threshold':<15} {'Accuracy':<10} {'Performance':<15}")
    print("-" * 50)

    max_acc = max(accuracies)
    max_idx = accuracies.index(max_acc)

    for i, (threshold, acc) in enumerate(zip(thresholds, accuracies)):
        performance = ""
        if i == max_idx:
            performance = "← BEST"
        elif acc > 0.7:
            performance = "Good"
        elif acc > 0.5:
            performance = "Moderate"
        else:
            performance = "Poor"

        print(f"{threshold:<15.1f} {acc:<10.4f} {performance:<15}")

    print("-" * 50)
    print(f"Best IoU threshold: {thresholds[max_idx]} (Accuracy: {max_acc:.4f})")
    print("=" * 50)


def main():
    x = [i / 10.0 for i in range(1, 10)]
    accuracies = []

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

    print(f"\nTesting {len(x)} different IoU thresholds...")
    print("=" * 60)

    for i, threshold in enumerate(x):
        thisAccuracies = []
        print(f"\nTesting IoU threshold: {threshold}")

        for j in range(3):  # Run multiple times for averaging
            print(f"  Run {j + 1}/3...", end=" ")

            # Show image predictions only for first threshold and first run
            show_images = i == 0 and j == 0
            if show_images:
                print("\n  [Showing image predictions for first 5 samples]")
                print("  Close each image window to continue...")

            accuracy = calculate_accuracy_with_iou_threshold(
                model, test_loader, device, threshold, show_images=False
            )
            thisAccuracies.append(accuracy)

            if not show_images:
                print(f"  Accuracy: {accuracy:.4f}")

        avg_accuracy = statistics.mean(thisAccuracies)
        accuracies.append(avg_accuracy)
        print(f"  Average accuracy: {avg_accuracy:.4f}")

    # Print detailed results
    print_detailed_results(x, accuracies)

    # Plot results
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({"font.size": 16})
    plt.rcParams["savefig.directory"] = os.path.expanduser(
        "~/Programming/RobotMower/finalReport/images/"
    )

    plt.scatter(x, accuracies, color="blue", marker="o", s=100, alpha=0.7)
    plt.xlabel("IoU Threshold")
    plt.ylabel("Detection Accuracy")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
