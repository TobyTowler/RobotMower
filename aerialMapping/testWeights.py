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

from maskRCNNmodel import GolfCourseDataset, get_transform, get_model


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


def predict_image(model, image_path, class_names, confidence_threshold=0.7):
    model.eval()
    device = next(model.parameters()).device

    image = Image.open(image_path).convert("RGB")
    to_tensor = torchvision.transforms.ToTensor()
    image_tensor = to_tensor(image).to(device)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    image_np = image_tensor.cpu().permute(1, 2, 0).numpy()

    result = {"image": image_np, "boxes": [], "masks": [], "classes": [], "scores": []}

    for box, mask, label, score in zip(
        prediction["boxes"],
        prediction["masks"],
        prediction["labels"],
        prediction["scores"],
    ):
        if score >= confidence_threshold:
            result["boxes"].append(box.cpu().numpy())
            result["masks"].append(mask[0].cpu().numpy() > 0.5)
            result["classes"].append(class_names[label])
            result["scores"].append(score.cpu().numpy())

    return result


def visualize_prediction(result, output_path=None, show=True):
    image = result["image"].copy()

    color_map = {
        "background": (0, 0, 0),
        "green": (0, 200, 0),
        "fairway": (0, 100, 0),
        "bunker": (255, 255, 150),
        "rough": (100, 150, 0),
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
    model, image_dir, class_names, output_dir=None, confidence_threshold=0.7
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
    model, image_dir, class_names, output_dir=None, confidence_threshold=0.7
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


def genMap(img):
    model_path = "models/golf_course_model_best.pth"

    class_names = ["background", "green", "fairway", "bunker", "rough", "water"]

    print(f"Loading model from {model_path}...")

    model = load_model(model_path, 6)

    print("\nProcessing individual image...")

    test_image = img
    result = predict_image(model, test_image, class_names, confidence_threshold=0.6)
    visualize_prediction(result, "single_prediction.png", show=True)


if __name__ == "__main__":
    genMap("./imgs/rawImgs/Benniksgaard_Golf_Klub_1000_02_1.jpg")
