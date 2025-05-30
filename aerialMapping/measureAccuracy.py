import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
import traceback

from testWeights import load_model
from maskRCNNmodel import GolfCourseDataset, get_transform, get_model


def calculate_mask_iou(pred_mask, true_mask, threshold=0.5):
    if pred_mask.sum() == 0 and true_mask.sum() == 0:
        return 1.0

    pred_binary = (pred_mask > threshold).float()
    true_binary = true_mask.float()

    intersection = (pred_binary * true_binary).sum()
    union = pred_binary.sum() + true_binary.sum() - intersection

    if union == 0:
        return 0.0

    return (intersection / union).item()


def evaluate_model_comprehensive(
    model, data_loader, device, iou_threshold=0.5, conf_threshold=0.5
):
    model.eval()

    class_names = ["background", "green", "fairway", "bunker", "rough", "water"]

    per_class_ious = {i: [] for i in range(len(class_names))}
    per_class_detection_metrics = {
        i: {"tp": 0, "fp": 0, "fn": 0} for i in range(len(class_names))
    }

    total_pixel_correct = 0
    total_pixels = 0

    print("Starting evaluation...")

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx + 1}/{len(data_loader)}")

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)

            for pred, target in zip(predictions, targets):
                img_height, img_width = images[0].shape[-2:]

                if len(pred["scores"]) > 0:
                    high_conf_indices = pred["scores"] > conf_threshold
                    pred_boxes = pred["boxes"][high_conf_indices]
                    pred_labels = pred["labels"][high_conf_indices]
                    pred_masks = pred["masks"][high_conf_indices]
                    pred_scores = pred["scores"][high_conf_indices]
                else:
                    pred_boxes = torch.empty((0, 4))
                    pred_labels = torch.empty(0, dtype=torch.long)
                    pred_masks = torch.empty((0, img_height, img_width))
                    pred_scores = torch.empty(0)

                true_boxes = target["boxes"]
                true_labels = target["labels"]
                true_masks = target["masks"]

                if len(pred_boxes) > 0 and len(true_boxes) > 0:
                    box_ious = box_iou(pred_boxes, true_boxes)

                    matched_pred = set()
                    matched_true = set()

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
                            per_class_detection_metrics[true_label_idx]["tp"] += 1
                            matched_pred.add(best_pred_idx)
                            matched_true.add(i)

                            mask_iou = calculate_mask_iou(
                                pred_masks[best_pred_idx].squeeze(),
                                true_masks[i].float(),
                            )
                            per_class_ious[true_label_idx].append(mask_iou)
                        else:
                            per_class_detection_metrics[true_label_idx]["fn"] += 1

                    for j, pred_label in enumerate(pred_labels):
                        if j not in matched_pred:
                            per_class_detection_metrics[pred_label.item()]["fp"] += 1

                elif len(true_boxes) > 0:
                    for true_label in true_labels:
                        per_class_detection_metrics[true_label.item()]["fn"] += 1

                elif len(pred_boxes) > 0:
                    for pred_label in pred_labels:
                        per_class_detection_metrics[pred_label.item()]["fp"] += 1

                if len(true_masks) > 0:
                    combined_true_mask = torch.zeros(
                        (img_height, img_width), device=device
                    )
                    for i, mask in enumerate(true_masks):
                        label_val = true_labels[i].item()
                        combined_true_mask[mask.bool()] = label_val

                    combined_pred_mask = torch.zeros(
                        (img_height, img_width), device=device
                    )
                    if len(pred_masks) > 0:
                        for i, mask in enumerate(pred_masks):
                            label_val = pred_labels[i].item()
                            mask_binary = mask.squeeze() > 0.5
                            combined_pred_mask[mask_binary] = label_val

                    correct_pixels = (
                        (combined_true_mask == combined_pred_mask).sum().item()
                    )
                    total_pixel_correct += correct_pixels
                    total_pixels += img_height * img_width

    print("Calculating final metrics...")

    results = {}

    for class_idx in range(len(class_names)):
        metrics = per_class_detection_metrics[class_idx]
        tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        mean_iou = (
            np.mean(per_class_ious[class_idx]) if per_class_ious[class_idx] else 0
        )

        results[f"{class_names[class_idx]}_precision"] = precision
        results[f"{class_names[class_idx]}_recall"] = recall
        results[f"{class_names[class_idx]}_f1"] = f1
        results[f"{class_names[class_idx]}_iou"] = mean_iou
        results[f"{class_names[class_idx]}_tp"] = tp
        results[f"{class_names[class_idx]}_fp"] = fp
        results[f"{class_names[class_idx]}_fn"] = fn

    all_tp = sum(per_class_detection_metrics[i]["tp"] for i in range(len(class_names)))
    all_fp = sum(per_class_detection_metrics[i]["fp"] for i in range(len(class_names)))
    all_fn = sum(per_class_detection_metrics[i]["fn"] for i in range(len(class_names)))

    overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    overall_f1 = (
        2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0
    )

    all_ious = []
    for class_idx in range(1, len(class_names)):
        if per_class_ious[class_idx]:
            all_ious.extend(per_class_ious[class_idx])
    mean_iou = np.mean(all_ious) if all_ious else 0

    pixel_accuracy = total_pixel_correct / total_pixels if total_pixels > 0 else 0

    results.update(
        {
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1": overall_f1,
            "mean_iou": mean_iou,
            "pixel_accuracy": pixel_accuracy,
            "total_detections": all_tp + all_fp,
            "total_ground_truth": all_tp + all_fn,
        }
    )

    return results


def measure_model_accuracy(
    model_path="./models/golf_course_model_best.pth",
    img_dir="./imgs/rawImgs/",
    json_dir="./imgs/annotationsTesting/",
    confidence_threshold=0.5,
    iou_threshold=0.5,
):
    print("GOLF COURSE MODEL ACCURACY MEASUREMENT")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    try:
        print("\nLoading dataset...")
        test_dataset = GolfCourseDataset(
            img_dir=img_dir,
            json_dir=json_dir,
            transforms=get_transform(train=False),
            compute_class_weights=False,
        )

        print(f"Dataset loaded: {len(test_dataset)} samples")

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: tuple(zip(*x)),
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        print(f"\nLoading model...")
        model = load_model(model_path, len(test_dataset.class_names))
        model.to(device)

        print(f"\nRunning evaluation on {len(test_dataset)} samples...")
        results = evaluate_model_comprehensive(
            model,
            test_loader,
            device,
            iou_threshold=iou_threshold,
            conf_threshold=confidence_threshold,
        )

        print_evaluation_results(results)

        return results

    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()
        return None


def quick_accuracy_check(model_path="./models/golf_course_model_best.pth"):
    return measure_model_accuracy(model_path)


if __name__ == "__main__":
    print("Running basic accuracy measurement...")
    results = quick_accuracy_check()

    if results:
        print(f"\nQuick Summary:")
        print(f"Overall F1-Score: {results['overall_f1']:.4f}")
        print(f"Mean IoU: {results['mean_iou']:.4f}")
        print(f"Pixel Accuracy: {results['pixel_accuracy']:.4f}")
