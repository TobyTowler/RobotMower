import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os


def load_model_for_debug(model_path):
    # Load state dict
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    # Check number of classes
    num_classes = state_dict["roi_heads.box_predictor.cls_score.bias"].shape[0]
    print(f"Model has {num_classes} output classes")

    # Create model with correct number of classes
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    # Load weights
    model.load_state_dict(state_dict)
    model.eval()

    return model, num_classes


def run_debug_detection(model, image_path, class_names):
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    transform = torchvision.transforms.ToTensor()
    image_tensor = transform(image)

    # Run inference with no threshold
    with torch.no_grad():
        prediction = model([image_tensor])[0]

    # Get all predictions
    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    masks = prediction["masks"].cpu().numpy()

    # Print all predictions sorted by confidence
    print("\nAll predictions with confidence scores:")

    # Create indices for sorting by score
    sorted_indices = np.argsort(scores)[::-1]  # Descending order

    for i in sorted_indices:
        label_idx = labels[i]
        class_name = (
            class_names[label_idx]
            if label_idx < len(class_names)
            else f"Unknown ({label_idx})"
        )
        print(f"  {class_name}: {scores[i]:.4f} - Box: {boxes[i]}")

    # Visualize top predictions by class
    visualize_top_predictions(image_tensor, boxes, labels, scores, masks, class_names)

    return boxes, labels, scores, masks


def visualize_top_predictions(
    image, boxes, labels, scores, masks, class_names, threshold=0.01
):
    """Visualize predictions with very low threshold"""
    # Convert image for display
    image_np = image.permute(1, 2, 0).numpy()

    # Define colors for each class
    colors = {
        "background": (0, 0, 0),  # Black
        "green": (0, 200, 0),  # Green
        "fairway": (0, 100, 0),  # Dark Green
        "bunker": (255, 255, 150),  # Light Yellow
        "rough": (100, 150, 0),  # Olive Green
        "water": (0, 100, 255),  # Blue
    }

    # Create figure with class-specific subplots
    unique_classes = np.unique(labels)
    if len(unique_classes) > 0:
        fig, axes = plt.subplots(
            1, len(unique_classes), figsize=(6 * len(unique_classes), 6)
        )
        if len(unique_classes) == 1:
            axes = [axes]  # Ensure axes is iterable for a single subplot

        for ax_idx, class_idx in enumerate(unique_classes):
            if class_idx >= len(class_names):
                class_name = f"Unknown ({class_idx})"
            else:
                class_name = class_names[class_idx]

            axes[ax_idx].imshow(image_np)
            axes[ax_idx].set_title(f"Class: {class_name}")

            # Get indices for this class
            class_mask = labels == class_idx
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]

            # Draw boxes for this class
            for box, score in zip(class_boxes, class_scores):
                if score > threshold:
                    x1, y1, x2, y2 = box.astype(int)
                    color = colors.get(class_name, (255, 0, 0))
                    color_norm = [c / 255 for c in color]

                    rect = plt.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        edgecolor=color_norm,
                        linewidth=2,
                    )
                    axes[ax_idx].add_patch(rect)

                    # Add score text
                    axes[ax_idx].text(
                        x1,
                        y1 - 5,
                        f"{score:.2f}",
                        color="white",
                        fontsize=12,
                        bbox=dict(facecolor=color_norm, alpha=0.7),
                    )

            axes[ax_idx].axis("off")

        plt.tight_layout()
        plt.savefig("debug_detection_by_class.png")
        plt.show()
    else:
        print("No classes detected in the image.")


def check_dataset_balance(json_dir):
    """Check for class imbalance in dataset"""
    class_counts = {
        "background": 0,
        "green": 0,
        "fairway": 0,
        "bunker": 0,
        "rough": 0,
        "water": 0,
    }

    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

    for json_file in json_files:
        try:
            with open(os.path.join(json_dir, json_file), "r") as f:
                import json

                data = json.load(f)

                for shape in data.get("shapes", []):
                    if shape.get("shape_type") == "polygon":
                        label = shape.get("label", "")
                        if label in class_counts:
                            class_counts[label] += 1
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    print("\nClass distribution in dataset:")
    total = sum(class_counts.values())
    for cls, count in class_counts.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {cls}: {count} instances ({percentage:.1f}%)")

    # Check for severe imbalance
    if total > 0:
        max_class = max(class_counts.items(), key=lambda x: x[1])
        if max_class[1] / total > 0.5:  # One class has more than 50% of instances
            print(
                f"\nWARNING: Severe class imbalance detected. {max_class[0]} makes up {max_class[1] / total * 100:.1f}% of annotations."
            )
            print("This could explain why your model is only detecting one class.")


def main():
    # Settings
    model_path = "./models/golf_course_model_best.pth"
    test_image = "./imgs/testingdata/Benniksgaard_Golf_Klub_1000_02_2.jpg"
    class_names = ["background", "green", "fairway", "bunker", "rough", "water"]

    # Check dataset balance
    json_dir = "./imgs/annotations/"
    check_dataset_balance(json_dir)

    # Load model
    model, num_classes = load_model_for_debug(model_path)

    # Verify class count matches
    if num_classes != len(class_names):
        print(
            f"WARNING: Model has {num_classes} classes but {len(class_names)} class names provided!"
        )
        print("This mismatch could be causing your issue.")

    # Run detection with debug info
    run_debug_detection(model, test_image, class_names)

    # Check model weights to see if one class dominates
    cls_weights = model.roi_heads.box_predictor.cls_score.weight
    cls_bias = model.roi_heads.box_predictor.cls_score.bias

    print("\nClass prediction bias values:")
    for i, bias in enumerate(cls_bias):
        class_name = class_names[i] if i < len(class_names) else f"Unknown ({i})"
        print(f"  {class_name}: {bias.item():.4f}")

    # Plot weight distributions to check for anomalies
    plt.figure(figsize=(10, 6))
    for i in range(num_classes):
        class_name = class_names[i] if i < len(class_names) else f"Unknown ({i})"
        plt.hist(
            cls_weights[i].detach().cpu().numpy(), alpha=0.5, bins=30, label=class_name
        )

    plt.title("Weight Distributions by Class")
    plt.legend()
    plt.savefig("class_weight_distributions.png")
    plt.show()


if __name__ == "__main__":
    main()
