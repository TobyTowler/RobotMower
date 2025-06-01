import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import traceback

# Remove the old import - we'll define get_model here to match the improved training
# from maskRCNNmodel import GolfCourseDataset, get_transform, get_model


def get_model(num_classes):
    """IMPROVED model configuration matching the new training setup"""
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    # CRITICAL: Match the improved training configuration
    model.roi_heads.score_thresh = 0.3  # Lower threshold for rough detection
    model.roi_heads.nms_thresh = 0.3  # Lower NMS threshold
    model.roi_heads.detections_per_img = 200  # More detections per image

    return model


def load_model(model_path, num_classes):
    print(f"Loading model from {model_path}...")
    model = get_model(num_classes)

    checkpoint = torch.load(
        model_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    if "model_state_dict" in checkpoint:
        print("Loading from checkpoint with model_state_dict...")
        model.load_state_dict(checkpoint["model_state_dict"])

        if "epoch" in checkpoint:
            print(f"Model trained for {checkpoint['epoch'] + 1} epochs")
        if "val_loss" in checkpoint:
            print(f"Validation loss: {checkpoint['val_loss']:.4f}")
        if "class_accuracy" in checkpoint:
            print(f"Class accuracies: {checkpoint['class_accuracy']}")
    else:
        print("Loading from direct state_dict checkpoint...")
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def predict_image(model, image_path, class_names, confidence_threshold=0.3):
    model.eval()
    device = next(model.parameters()).device

    image = Image.open(image_path).convert("RGB")
    to_tensor = torchvision.transforms.ToTensor()
    image_tensor = to_tensor(image).to(device)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    image_np = image_tensor.cpu().permute(1, 2, 0).numpy()

    result = {"image": image_np, "boxes": [], "masks": [], "classes": [], "scores": []}

    print(f"Raw predictions found: {len(prediction['scores'])}")

    # Debug: Show all predictions
    for i, (box, mask, label, score) in enumerate(
        zip(
            prediction["boxes"],
            prediction["masks"],
            prediction["labels"],
            prediction["scores"],
        )
    ):
        class_name = class_names[label.item()]
        print(f"  {i}: {class_name} - {score:.3f}")

        if score >= confidence_threshold:
            result["boxes"].append(box.cpu().numpy())
            result["masks"].append(mask[0].cpu().numpy() > 0.5)
            result["classes"].append(class_name)
            result["scores"].append(score.cpu().numpy())

    print(f"Filtered predictions (>= {confidence_threshold}): {len(result['boxes'])}")

    # Special debug for rough
    rough_predictions = [
        (class_names[label.item()], score.item())
        for label, score in zip(prediction["labels"], prediction["scores"])
        if class_names[label.item()] == "rough"
    ]
    print(f"Rough predictions found: {len(rough_predictions)}")
    for cls, score in rough_predictions:
        print(f"  - {cls}: {score:.3f}")

    return result


def visualize_prediction(result, output_path=None, show=True):
    image = result["image"].copy()

    color_map = {
        "background": (0, 0, 0),
        "green": (0, 200, 0),
        "fairway": (0, 100, 0),
        "bunker": (255, 255, 150),
        "rough": (150, 75, 0),  # Changed to brown/orange for better visibility
        "water": (0, 100, 255),
    }

    overlay = np.zeros_like(image)

    for mask, class_name in zip(result["masks"], result["classes"]):
        color = color_map.get(class_name, (255, 0, 0))
        mask_rgb = np.zeros_like(image)
        mask_rgb[mask] = color
        overlay[mask] = color

    alpha = 0.4
    blended = cv2.addWeighted(image * 255, 1 - alpha, overlay, alpha, 0)

    plt.figure(figsize=(12, 10))
    plt.imshow(blended / 255)

    for box, class_name, score in zip(
        result["boxes"], result["classes"], result["scores"]
    ):
        x1, y1, x2, y2 = box.astype(int)
        color = color_map.get(class_name, (255, 0, 0))
        plt.gca().add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=[c / 255 for c in color],
                linewidth=2,
            )
        )
        plt.text(
            x1,
            y1 - 10,
            f"{class_name}: {score:.2f}",
            color="white",
            bbox=dict(facecolor=[c / 255 for c in color], alpha=0.8),
        )

    plt.title("Golf Course Feature Detection")
    plt.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)

    if show:
        plt.show()
    else:
        plt.close()

    return blended


def test_model_on_directory(
    model, image_dir, class_names, output_dir=None, confidence_threshold=0.3
):
    os.makedirs(output_dir, exist_ok=True)

    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)

        print(f"Processing {image_file}...")

        result = predict_image(model, image_path, class_names, confidence_threshold)

        if output_dir:
            output_path = os.path.join(output_dir, f"pred_{image_file}")
            visualize_prediction(result, output_path, show=False)

            outline_image = create_outline_image(
                result, output_path.replace(".", "_outlines.")
            )

        print(f"Found {len(result['boxes'])} features")


def create_outline_image(result, output_path=None):
    image = result["image"].copy() * 255

    color_map = {
        "background": (0, 0, 0),
        "green": (0, 200, 0),
        "fairway": (0, 100, 0),
        "bunker": (255, 255, 150),
        "rough": (100, 150, 0),
        "water": (0, 100, 255),
    }

    outlines = image.copy()

    for mask, class_name in zip(result["masks"], result["classes"]):
        color = color_map.get(class_name, (255, 0, 0))

        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(outlines, contours, -1, color, 2)

    if output_path:
        plt.figure(figsize=(12, 10))
        plt.imshow(outlines / 255)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    return outlines


def process_and_show_all_images(
    model, image_dir, class_names, output_dir=None, confidence_threshold=0.3
):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' does not exist.")
        return

    image_files = []
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
    for f in os.listdir(image_dir):
        if f.lower().endswith(valid_extensions):
            image_files.append(f)

    if not image_files:
        print(f"No image files found in '{image_dir}'.")
        return

    print(f"Found {len(image_files)} images to process.")

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"\nProcessing {image_file}...")

        try:
            if not os.path.isfile(image_path):
                print(f"ERROR: File does not exist.")
                continue

            result = predict_image(model, image_path, class_names, confidence_threshold)

            print(f"Found {len(result['boxes'])} features:")
            for cls, score in zip(result["classes"], result["scores"]):
                print(f"  - {cls}: {score:.2f}")

            if output_dir:
                output_path = os.path.join(output_dir, f"pred_{image_file}")
            else:
                output_path = None

            visualize_prediction(result, output_path, show=True)

            if output_dir and len(result["boxes"]) > 0:
                outline_path = os.path.join(output_dir, f"outline_{image_file}")
                create_outline_image(result, outline_path)

        except Exception as e:
            print(f"ERROR: Failed to process: {str(e)}")
            traceback.print_exc()

    print("\nAll images processed.")


def debug_all_predictions(img):
    """Debug function to see ALL predictions regardless of confidence"""
    model_path = "models/golf_course_model_best.pth"
    class_names = ["background", "green", "fairway", "bunker", "rough", "water"]

    model = load_model(model_path, 6)

    print("\n=== DEBUGGING ALL PREDICTIONS ===")
    result = predict_image(model, img, class_names, confidence_threshold=0.1)

    print(f"\nSUMMARY:")
    print(f"Total detections (>0.1): {len(result['boxes'])}")

    class_counts = {}
    for cls in result["classes"]:
        class_counts[cls] = class_counts.get(cls, 0) + 1

    for cls, count in class_counts.items():
        print(f"  {cls}: {count} detections")

    return result


def genMap(img):
    model_path = "models/golf_course_model_best.pth"
    class_names = ["background", "green", "fairway", "bunker", "rough", "water"]

    model = load_model(model_path, 6)

    print("\nProcessing individual image...")

    # Lower confidence threshold for rough detection
    result = predict_image(model, img, class_names, confidence_threshold=0.3)
    visualize_prediction(result, "single_prediction.png", show=True)

    return result


if __name__ == "__main__":
    # Debug version to see what's happening
    test_image = "./imgs/rawImgs/Benniksgaard_Golf_Klub_1000_02_1.jpg"

    print("=== DEBUGGING MODE ===")
    debug_result = debug_all_predictions(test_image)

    print("\n=== NORMAL PREDICTION ===")
    normal_result = genMap(test_image)
