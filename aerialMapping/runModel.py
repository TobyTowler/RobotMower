import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from aerialMapping.maskRCNNmodel import get_model
# from maskRCNNmodel import get_model
# from aerialMapping.testWeights import visualize_prediction


def display_outlines(image_path, results, output_path=None, show=True):
    # Load image
    image = np.array(Image.open(image_path).convert("RGB"))

    # Color map matching your visualize_prediction function
    color_map = {
        "background": (0, 0, 0),
        "green": (0, 200, 0),
        "fairway": (0, 100, 0),
        "bunker": (255, 255, 150),
        "rough": (100, 150, 0),
        "water": (0, 100, 255),
    }

    # Create overlay
    overlay = np.zeros_like(image)

    # Fill outline regions
    for result in results:
        class_name = result["class"]
        points = np.array(result["outline_points"], dtype=np.int32)
        color = color_map.get(class_name, (255, 0, 0))
        cv2.fillPoly(overlay, [points], color)

    # Blend with same alpha as original
    alpha = 0.4
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    # Display with matplotlib like original
    plt.figure(figsize=(12, 10))
    plt.imshow(blended / 255)

    # Add labels
    for result in results:
        points = np.array(result["outline_points"], dtype=np.int32)
        class_name = result["class"]
        confidence = result["confidence"]
        color = color_map.get(class_name, (255, 0, 0))

        # Get bounding box for text placement
        x, y, w, h = cv2.boundingRect(points)

        plt.text(
            x,
            y - 10,
            f"{class_name}: {confidence:.2f}",
            color="white",
            bbox=dict(facecolor=[c / 255 for c in color], alpha=0.8),
        )

    # for i, result in enumerate(results):
    #     points = result["outline_points"]
    #     class_name = result["class"]
    #
    #     # Mark first 2 points with different colors
    #     if len(points) >= 1:
    #         plt.scatter(
    #             points[0][0],
    #             points[0][1],
    #             c="yellow",
    #             s=200,
    #             marker="o",
    #             edgecolors="black",
    #             linewidth=2,
    #             label=f"{class_name} Point 1" if i == 0 else "",
    #         )
    #     if len(points) >= 2:
    #         plt.scatter(
    #             points[1][0],
    #             points[1][1],
    #             c="orange",
    #             s=200,
    #             marker="s",
    #             edgecolors="black",
    #             linewidth=2,
    #             label=f"{class_name} Point 2" if i == 0 else "",
    #         )
    #
    # plt.legend()

    plt.title("Golf Course Feature Detection")
    plt.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

    return blended


def run_model_and_get_outlines(image_path):
    class_names = ["background", "green", "fairway", "bunker", "rough", "water"]

    model_path = "aerialMapping/models/golf_course_model_best1.pth"
    # model_path = "models/golf_course_model_best.pth"
    print(f"Loading model from {model_path}...")
    model = get_model(6)

    checkpoint = torch.load(
        model_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    device = next(model.parameters()).device

    image = Image.open(image_path).convert("RGB")
    to_tensor = torchvision.transforms.ToTensor()
    image_tensor = to_tensor(image).to(device)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    results = []

    for box, mask, label, score in zip(
        prediction["boxes"],
        prediction["masks"],
        prediction["labels"],
        prediction["scores"],
    ):
        if score >= 0.6:
            mask_np = mask[0].cpu().numpy() > 0.5
            mask_uint8 = (mask_np * 255).astype(np.uint8)

            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)

                epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

                outline_points = []
                for point in simplified_contour:
                    x, y = point[0]
                    outline_points.append([int(x), int(y)])

                results.append(
                    {
                        "class": class_names[label.item()],
                        "confidence": float(score.cpu().numpy()),
                        "outline_points": outline_points,
                    }
                )

    # display_outlines(image_path, results)

    return results


def show_first_two_vertices_of_first_detection(image_path, results, zoom_padding=50):
    # Load image
    image = np.array(Image.open(image_path).convert("RGB"))

    plt.figure(figsize=(12, 10))
    plt.imshow(image)

    # Only process the first detection
    if results:
        result = results[0]  # Get only the first detection
        points = result["outline_points"]
        class_name = result["class"]
        confidence = result["confidence"]

        print(f"\nFirst detection: {class_name} (confidence: {confidence:.2f})")

        # Show first vertex
        if len(points) >= 1:
            x1, y1 = points[0]
            plt.scatter(
                x1,
                y1,
                c="red",
                s=300,
                marker="o",
                edgecolors="black",
                linewidth=3,
                zorder=5,
            )
            plt.annotate(
                f"Vertex 1: ({x1}, {y1})",
                (x1, y1),
                xytext=(20, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="red", alpha=0.9),
                fontsize=12,
                color="white",
                weight="bold",
            )
            print(f"  Vertex 1: ({x1}, {y1})")

        # Show second vertex
        if len(points) >= 2:
            x2, y2 = points[1]
            plt.scatter(
                x2,
                y2,
                c="blue",
                s=300,
                marker="s",
                edgecolors="black",
                linewidth=3,
                zorder=5,
            )
            plt.annotate(
                f"Vertex 2: ({x2}, {y2})",
                (x2, y2),
                xytext=(20, -30),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="blue", alpha=0.9),
                fontsize=12,
                color="white",
                weight="bold",
            )
            print(f"  Vertex 2: ({x2}, {y2})")

        # Draw a line between first 2 vertices if both exist
        if len(points) >= 2:
            plt.plot(
                [points[0][0], points[1][0]],
                [points[0][1], points[1][1]],
                "purple",
                linewidth=4,
                alpha=0.8,
                zorder=4,
            )

        # Also draw the complete outline in light gray for context
        if len(points) > 2:
            outline_x = [p[0] for p in points] + [points[0][0]]  # Close the shape
            outline_y = [p[1] for p in points] + [points[0][1]]
            plt.plot(
                outline_x, outline_y, "gray", linewidth=2, alpha=0.5, linestyle="--"
            )

        # ZOOM IN on the first 2 vertices
        if len(points) >= 2:
            # Get coordinates of first 2 points
            x_coords = [points[0][0], points[1][0]]
            y_coords = [points[0][1], points[1][1]]

            # Calculate zoom boundaries
            min_x = min(x_coords) - zoom_padding
            max_x = max(x_coords) + zoom_padding
            min_y = min(y_coords) - zoom_padding
            max_y = max(y_coords) + zoom_padding

            # Set the plot limits to zoom in
            plt.xlim(min_x, max_x)
            plt.ylim(max_y, min_y)  # Note: reversed for image coordinates

        elif len(points) >= 1:
            # If only one point, zoom around it
            x1, y1 = points[0]
            plt.xlim(x1 - zoom_padding, x1 + zoom_padding)
            plt.ylim(y1 + zoom_padding, y1 - zoom_padding)

    else:
        print("No detections found!")

    plt.title("Zoomed View: First 2 Vertices of First Detection")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    image_path = "./imgs/rawImgs/Benniksgaard_Golf_Klub_1000_02_1.jpg"
    # image_path = "./imgs/rawImgs/Benniksgaard_Golf_Klub_1000_010.jpg"
    try:
        outlines = run_model_and_get_outlines(image_path)

        print(f"Found {len(outlines)} detections:")
        for i, detection in enumerate(outlines):
            print(
                f"{i + 1}. {detection['class']} (confidence: {detection['confidence']:.2f})"
            )
            print(f"   Outline points: {len(detection['outline_points'])} points")

            if detection["outline_points"]:
                print(f"   First few points: {detection['outline_points'][:3]}...")

        display_outlines(image_path, outlines)
        show_first_two_vertices_of_first_detection(
            image_path, outlines, zoom_padding=350
        )
        show_first_two_vertices_of_first_detection(image_path, outlines)
        return outlines

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    main()
