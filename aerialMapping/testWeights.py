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

# Your dataset class (from training script)
from maskRCNNmodel import GolfCourseDataset, get_transform, get_model


def load_model(model_path, num_classes):
    """Load a saved model from disk."""
    print(f"Loading model from {model_path}...")
    model = get_model(num_classes)

    # Load the checkpoint
    checkpoint = torch.load(
        model_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Check if this is a new-style checkpoint (with model_state_dict)
    if "model_state_dict" in checkpoint:
        print("Loading from checkpoint with model_state_dict...")
        model.load_state_dict(checkpoint["model_state_dict"])

        # Print any additional info if available
        if "epoch" in checkpoint:
            print(f"Model trained for {checkpoint['epoch'] + 1} epochs")
        if "val_loss" in checkpoint:
            print(f"Validation loss: {checkpoint['val_loss']:.4f}")
        if "class_accuracy" in checkpoint:
            print(f"Class accuracies: {checkpoint['class_accuracy']}")
    else:
        # Fall back to direct loading for old-style checkpoints
        print("Loading from direct state_dict checkpoint...")
        model.load_state_dict(checkpoint)

    model.eval()  # Set to evaluation mode
    return model


def predict_image(model, image_path, class_names, confidence_threshold=0.7):
    """Run inference on a single image"""
    # Set model to evaluation mode
    model.eval()
    device = next(model.parameters()).device

    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    to_tensor = torchvision.transforms.ToTensor()
    image_tensor = to_tensor(image).to(device)

    # Run inference
    with torch.no_grad():
        prediction = model([image_tensor])[0]

    # Convert back to numpy for visualization
    image_np = image_tensor.cpu().permute(1, 2, 0).numpy()

    # Process predictions
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
    """Visualize prediction results with colors for each class"""
    image = result["image"].copy()

    # Define colors for each class (using a color map)
    color_map = {
        "background": (0, 0, 0),  # Black
        "green": (0, 200, 0),  # Green
        "fairway": (0, 100, 0),  # Dark Green
        "bunker": (255, 255, 150),  # Light Yellow
        "rough": (100, 150, 0),  # Olive Green
        "water": (0, 100, 255),  # Blue
    }

    # Create a blank overlay to show all masks
    overlay = np.zeros_like(image)

    # Add each mask to the overlay
    for mask, class_name in zip(result["masks"], result["classes"]):
        color = color_map.get(class_name, (255, 0, 0))  # Default to red
        mask_rgb = np.zeros_like(image)
        mask_rgb[mask] = color
        overlay[mask] = color

    # Blend original image with overlay
    alpha = 0.4
    blended = cv2.addWeighted(image * 255, 1 - alpha, overlay, alpha, 0)

    # Create figure
    plt.figure(figsize=(12, 10))
    plt.imshow(blended / 255)

    # Draw bounding boxes and labels
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
    """Test model on all images in a directory"""
    os.makedirs(output_dir, exist_ok=True)

    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)

        print(f"Processing {image_file}...")

        # Predict
        result = predict_image(model, image_path, class_names, confidence_threshold)

        # Visualize
        if output_dir:
            output_path = os.path.join(output_dir, f"pred_{image_file}")
            visualize_prediction(result, output_path, show=False)

            # Also save a version with just the outlines
            outline_image = create_outline_image(
                result, output_path.replace(".", "_outlines.")
            )

        print(f"Found {len(result['boxes'])} features")


def create_outline_image(result, output_path=None):
    """Create an image showing just the outlines of detected features"""
    image = result["image"].copy() * 255

    # Define colors for each class
    color_map = {
        "background": (0, 0, 0),  # Black
        "green": (0, 200, 0),  # Green
        "fairway": (0, 100, 0),  # Dark Green
        "bunker": (255, 255, 150),  # Light Yellow
        "rough": (100, 150, 0),  # Olive Green
        "water": (0, 100, 255),  # Blue
    }

    # Create a blank image for outlines
    outlines = image.copy()

    # Draw outlines for each mask
    for mask, class_name in zip(result["masks"], result["classes"]):
        color = color_map.get(class_name, (255, 0, 0))

        # Find contours
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw contours
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
    """Process all images in a directory and show each result"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Check if directory exists
    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' does not exist.")
        return

    # Get all image files
    image_files = []
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
    for f in os.listdir(image_dir):
        if f.lower().endswith(valid_extensions):
            image_files.append(f)

    if not image_files:
        print(f"No image files found in '{image_dir}'.")
        return

    print(f"Found {len(image_files)} images to process.")

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"\nProcessing {image_file}...")

        try:
            # Check if file exists
            if not os.path.isfile(image_path):
                print(f"ERROR: File does not exist.")
                continue

            # Process the image
            result = predict_image(model, image_path, class_names, confidence_threshold)

            # Display features found
            print(f"Found {len(result['boxes'])} features:")
            for cls, score in zip(result["classes"], result["scores"]):
                print(f"  - {cls}: {score:.2f}")

            # Visualize and show the result
            if output_dir:
                output_path = os.path.join(output_dir, f"pred_{image_file}")
            else:
                output_path = None

            # Always show the image
            visualize_prediction(result, output_path, show=True)

            # Create outline image
            if output_dir and len(result["boxes"]) > 0:
                outline_path = os.path.join(output_dir, f"outline_{image_file}")
                create_outline_image(result, outline_path)

        except Exception as e:
            print(f"ERROR: Failed to process: {str(e)}")
            traceback.print_exc()

    print("\nAll images processed.")


def genMap(img):
    model_path = "models/golf_course_model_best.pth"  # Path to your trained model

    # Define correct class names for your model
    # The model expects 6 classes (including background), but your list only has 5
    # Adding background as the first class since that's the standard for detection models
    class_names = ["background", "green", "fairway", "bunker", "rough", "water"]

    print(f"Loading model from {model_path}...")
    # Load model with auto-detection of class count
    model = load_model(model_path, 6)

    # Process all images in directory and show each one
    # process_and_show_all_images(
    #     model, test_image_dir, class_names, output_dir, confidence_threshold=0.7
    # )

    # Also process a specific image
    print("\nProcessing individual image...")
    # test_image = "./imgs/testingdata/Benniksgaard_Golf_Klub_1000_02_2.jpg"
    # test_image = "./imgs/rawImgs/Benniksgaard_Golf_Klub_1000_04_01.jpg"
    # test_image = "./imgs/testingdata/Benniksgaard_Golf_Klub_1000_02_2.jpg"
    test_image = img
    result = predict_image(model, test_image, class_names, confidence_threshold=0.6)
    visualize_prediction(
        result, "single_prediction.png", show=True
    )  # Set show=True to display the image


if __name__ == "__main__":
    genMap("./imgs/rawImgs/Benniksgaard_Golf_Klub_1000_02_1.jpg")
