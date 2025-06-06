import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches


def visualize_golf_detections(json_file, image_path=None):
    with open(json_file, "r") as f:
        data = json.load(f)

    color_map = {
        "bunker": "#D2B48C",
        "green": "#228B22",
        "fairway": "#32CD32",
        "rough": "#6B8E23",
        "water": "#4169E1",
        "background": "#808080",
    }

    fig, ax = plt.subplots(1, 1, figsize=(15, 12))

    if image_path:
        try:
            from PIL import Image

            img = Image.open(image_path)
            ax.imshow(img, alpha=0.7)
        except Exception as e:
            print(f"Could not load image: {e}")

    legend_elements = []
    classes_drawn = set()

    for i, detection in enumerate(data["detections"]):
        class_name = detection["class"]
        confidence = detection["confidence"]
        points = detection["outline_points"]

        if len(points) < 3:
            continue

        polygon_points = np.array(points)

        color = color_map.get(class_name, "#FF0000")

        polygon = Polygon(
            polygon_points, facecolor=color, edgecolor="black", alpha=0.6, linewidth=1.5
        )
        ax.add_patch(polygon)

        centroid_x = np.mean(polygon_points[:, 0])
        centroid_y = np.mean(polygon_points[:, 1])

        ax.text(
            centroid_x,
            centroid_y,
            f"{class_name}\n{confidence:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.8,
                edgecolor="black",
            ),
        )

        if class_name not in classes_drawn:
            legend_elements.append(
                mpatches.Patch(color=color, label=class_name.title())
            )
            classes_drawn.add(class_name)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))

    ax.set_xlabel("X Coordinates (pixels)", fontsize=12)
    ax.set_ylabel("Y Coordinates (pixels)", fontsize=12)
    ax.set_title(
        f"Golf Course Detection Results\n{len(data['detections'])} detections",
        fontsize=14,
    )

    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()

    print(f"\nDetection Summary:")
    print(f"Total detections: {len(data['detections'])}")
    class_counts = {}
    for detection in data["detections"]:
        class_name = detection["class"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")


def visualize_first_detection_vertices(json_file, image_path=None):
    """
    Visualize just the first 2 vertices of the first detection
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    if not data["detections"]:
        print("No detections found!")
        return

    first_detection = data["detections"][0]
    points = first_detection["outline_points"]
    class_name = first_detection["class"]
    confidence = first_detection["confidence"]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    if image_path:
        try:
            from PIL import Image

            img = Image.open(image_path)
            ax.imshow(img)
        except Exception as e:
            print(f"Could not load image: {e}")

    if len(points) > 2:
        polygon_points = np.array(points)
        polygon = Polygon(
            polygon_points,
            facecolor="lightblue",
            edgecolor="blue",
            alpha=0.3,
            linewidth=2,
        )
        ax.add_patch(polygon)

    if len(points) >= 1:
        x1, y1 = points[0]
        ax.scatter(
            x1,
            y1,
            c="red",
            s=300,
            marker="o",
            edgecolors="black",
            linewidth=3,
            zorder=10,
            label="Vertex 1",
        )
        ax.annotate(
            f"V1: ({x1}, {y1})",
            (x1, y1),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="red", alpha=0.9),
            fontsize=12,
            color="white",
            weight="bold",
        )

    if len(points) >= 2:
        x2, y2 = points[1]
        ax.scatter(
            x2,
            y2,
            c="blue",
            s=300,
            marker="s",
            edgecolors="black",
            linewidth=3,
            zorder=10,
            label="Vertex 2",
        )
        ax.annotate(
            f"V2: ({x2}, {y2})",
            (x2, y2),
            xytext=(20, -30),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="blue", alpha=0.9),
            fontsize=12,
            color="white",
            weight="bold",
        )

        ax.plot(
            [points[0][0], points[1][0]],
            [points[0][1], points[1][1]],
            "purple",
            linewidth=4,
            alpha=0.8,
            zorder=9,
            label="Connection",
        )

    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(
        f"First Detection: {class_name} (confidence: {confidence:.3f})\nFirst 2 Vertices"
    )
    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()

    print(f"First detection: {class_name}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Total vertices: {len(points)}")
    if len(points) >= 2:
        print(f"Vertex 1: {points[0]}")
        print(f"Vertex 2: {points[1]}")


if __name__ == "__main__":
    json_file = "../outputs/transformedOutlines/Benniksgaard_Golf_Klub_1000_02_1_outlines_transformed.json"
    image_file = "imgs/rawImgs/Benniksgaard_Golf_Klub_1000_02_1.jpg"

    print("Visualizing all detections...")
    visualize_golf_detections(json_file, image_file)

    print("\nVisualizing first detection vertices...")
    visualize_first_detection_vertices(json_file, image_file)
