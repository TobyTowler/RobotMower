import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches


def visualize_all_outlines(json_file, image_path=None):
    with open(json_file, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data['detections'])} detections")

    color_map = {
        "bunker": "#D2B48C",
        "green": "#228B22",
        "fairway": "#32CD32",
        "rough": "#6B8E23",
        "water": "#4169E1",
        "background": "#808080",
    }

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    if image_path:
        try:
            from PIL import Image

            img = Image.open(image_path)
            ax.imshow(img, alpha=0.5)
            print(f"Loaded background image: {img.size}")
        except Exception as e:
            print(f"Could not load image: {e}")

    legend_elements = []
    classes_drawn = set()
    drawn_count = 0

    for i, detection in enumerate(data["detections"]):
        class_name = detection["class"]
        confidence = detection["confidence"]
        points = detection["outline_points"]

        if len(points) < 3:
            print(f"Skipping {class_name} - only {len(points)} points")
            continue

        color = color_map.get(class_name, "#FF0000")

        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        ax.fill(
            x_coords, y_coords, color=color, alpha=0.4, edgecolor="black", linewidth=1.5
        )

        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])
        ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.8)

        center_x = np.mean([p[0] for p in points])
        center_y = np.mean([p[1] for p in points])

        if len(points) > 5:
            ax.text(
                center_x,
                center_y,
                f"{class_name}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
            )

        drawn_count += 1

        if class_name not in classes_drawn:
            legend_elements.append(
                mpatches.Patch(
                    color=color,
                    label=f"{class_name.title()} ({sum(1 for d in data['detections'] if d['class'] == class_name)})",
                )
            )
            classes_drawn.add(class_name)

    print(f"Drew {drawn_count} detection outlines")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0, 1))

    ax.set_xlabel("X Coordinates (pixels)", fontsize=12)
    ax.set_ylabel("Y Coordinates (pixels)", fontsize=12)
    ax.set_title(
        f"Golf Course Detection Outlines\n{drawn_count} of {len(data['detections'])} detections shown",
        fontsize=14,
        fontweight="bold",
    )

    if image_path:
        ax.invert_yaxis()

    plt.tight_layout()
    plt.show()

    print(f"\nDetection Summary:")
    class_counts = {}
    for detection in data["detections"]:
        class_name = detection["class"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")


def visualize_simple_outlines(json_file):
    """
    Very simple outline visualization without image background
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    fig, ax = plt.subplots(1, 1, figsize=(15, 12))

    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    for i, detection in enumerate(data["detections"]):
        points = detection["outline_points"]
        class_name = detection["class"]

        if len(points) >= 3:
            x_coords = [p[0] for p in points] + [points[0][0]]
            y_coords = [p[1] for p in points] + [points[0][1]]

            color = colors[i % len(colors)]

            ax.fill(x_coords[:-1], y_coords[:-1], color=color, alpha=0.4)

            ax.plot(x_coords, y_coords, color=color, linewidth=2, label=f"{class_name}")

            ax.scatter(points[0][0], points[0][1], color="black", s=50, zorder=10)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"All Detection Outlines ({len(data['detections'])} total)", fontsize=14
    )
    ax.set_xlabel("X Coordinates", fontsize=12)
    ax.set_ylabel("Y Coordinates", fontsize=12)

    plt.tight_layout()
    plt.show()


def show_detection_info(json_file):
    """
    Print detailed information about all detections
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    print(f"=== DETECTION INFO ===")
    print(f"Total detections: {len(data['detections'])}")

    all_x, all_y = [], []

    for i, detection in enumerate(data["detections"]):
        points = detection["outline_points"]
        class_name = detection["class"]
        confidence = detection["confidence"]

        if points:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            all_x.extend(x_coords)
            all_y.extend(y_coords)

            print(
                f"{i + 1:2d}. {class_name:8s} | {confidence:.3f} | {len(points):2d} points | X: {min(x_coords):4.0f}-{max(x_coords):4.0f} | Y: {min(y_coords):4.0f}-{max(y_coords):4.0f}"
            )

    if all_x and all_y:
        print(f"\nOverall bounds:")
        print(f"X: {min(all_x)} to {max(all_x)} (range: {max(all_x) - min(all_x)})")
        print(f"Y: {min(all_y)} to {max(all_y)} (range: {max(all_y) - min(all_y)})")


def plot_with_image(json_file, image_file):
    """Plot with background image"""
    print("=== PLOTTING WITH IMAGE BACKGROUND ===")
    show_detection_info(json_file)
    visualize_all_outlines(json_file, image_file)


def plot_without_image(json_file):
    """Plot without background image"""
    print("=== PLOTTING WITHOUT IMAGE BACKGROUND ===")
    show_detection_info(json_file)
    visualize_simple_outlines(json_file)


def plot_both(json_file, image_file=None):
    """Plot both versions"""
    show_detection_info(json_file)
    print("\n" + "=" * 50)
    print("SIMPLE OUTLINE PLOT")
    print("=" * 50)
    visualize_simple_outlines(json_file)

    if image_file:
        print("\n" + "=" * 50)
        print("PLOT WITH IMAGE BACKGROUND")
        print("=" * 50)
        visualize_all_outlines(json_file, image_file)


if __name__ == "__main__":
    json_file = "../outputs/transformedOutlines/Benniksgaard_Golf_Klub_1000_010_outlines_transformed.json"
    image_file = "../aerialMapping/imgs/rawImgs/Benniksgaard_Golf_Klub_1000_010.jpg"

    plot_both(json_file, image_file)
