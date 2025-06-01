import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches


def visualize_all_outlines(json_file, image_path=None):
    """
    Simple visualization of all detection outlines from JSON file
    """
    # Load the JSON data
    with open(json_file, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data['detections'])} detections")

    # Color mapping for different golf course features
    color_map = {
        "bunker": "#D2B48C",  # Sandy brown
        "green": "#228B22",  # Forest green
        "fairway": "#32CD32",  # Lime green
        "rough": "#6B8E23",  # Olive drab
        "water": "#4169E1",  # Royal blue
        "background": "#808080",  # Gray
    }

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Load and display background image if provided
    if image_path:
        try:
            from PIL import Image

            img = Image.open(image_path)
            ax.imshow(img, alpha=0.5)
            print(f"Loaded background image: {img.size}")
        except Exception as e:
            print(f"Could not load image: {e}")

    # Draw each detection outline
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

        # Get color for this class
        color = color_map.get(class_name, "#FF0000")

        # Draw outline as both fill and line
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        # Fill the polygon
        ax.fill(
            x_coords, y_coords, color=color, alpha=0.4, edgecolor="black", linewidth=1.5
        )

        # Draw outline
        # Close the polygon by adding first point at end
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])
        ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.8)

        # Add class label at center
        center_x = np.mean([p[0] for p in points])
        center_y = np.mean([p[1] for p in points])

        # Only add text if polygon is large enough
        if len(points) > 5:  # Only label larger polygons to avoid clutter
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

        # Add to legend if not already added
        if class_name not in classes_drawn:
            legend_elements.append(
                mpatches.Patch(
                    color=color,
                    label=f"{class_name.title()} ({sum(1 for d in data['detections'] if d['class'] == class_name)})",
                )
            )
            classes_drawn.add(class_name)

    print(f"Drew {drawn_count} detection outlines")

    # Set aspect ratio and styling
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Add legend
    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0, 1))

    # Labels and title
    ax.set_xlabel("X Coordinates (pixels)", fontsize=12)
    ax.set_ylabel("Y Coordinates (pixels)", fontsize=12)
    ax.set_title(
        f"Golf Course Detection Outlines\n{drawn_count} of {len(data['detections'])} detections shown",
        fontsize=14,
        fontweight="bold",
    )

    # Invert y-axis to match image coordinates if using image
    if image_path:
        ax.invert_yaxis()

    plt.tight_layout()
    plt.show()

    # Print summary
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
            # Close the polygon
            x_coords = [p[0] for p in points] + [points[0][0]]
            y_coords = [p[1] for p in points] + [points[0][1]]

            color = colors[i % len(colors)]

            # Draw filled polygon
            ax.fill(x_coords[:-1], y_coords[:-1], color=color, alpha=0.4)
            # Draw outline
            ax.plot(x_coords, y_coords, color=color, linewidth=2, label=f"{class_name}")

            # Mark first point with a dot
            ax.scatter(points[0][0], points[0][1], color="black", s=50, zorder=10)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"All Detection Outlines ({len(data['detections'])} total)", fontsize=14
    )
    ax.set_xlabel("X Coordinates", fontsize=12)
    ax.set_ylabel("Y Coordinates", fontsize=12)

    # Don't invert y-axis for simple plot
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


# Usage functions
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


# Usage examples:
if __name__ == "__main__":
    # Update these paths to match your files
    # json_file = "../outputs/transformedOutlines/Benniksgaard_Golf_Klub_1000_02_1_outlines_transformed.json"
    json_file = "../outputs/transformedOutlines/Benniksgaard_Golf_Klub_1000_010_outlines_transformed.json"
    image_file = (
        "../aerialMapping/imgs/rawImgs/Benniksgaard_Golf_Klub_1000_010.jpg"  # Optional
    )

    # Show all information and both plot types
    plot_both(json_file, image_file)

    # Or use individual functions:
    # show_detection_info(json_file)
    # plot_without_image(json_file)
    # plot_with_image(json_file, image_file)
