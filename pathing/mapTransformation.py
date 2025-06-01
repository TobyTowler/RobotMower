import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as patches


class SimpleGPSTransformer:
    def __init__(self, json_file_path):
        # Load the JSON file
        with open(json_file_path, "r") as f:
            self.data = json.load(f)

        # Find corner points from all detections
        all_points = []
        for detection in self.data["detections"]:
            all_points.extend(detection["outline_points"])

        all_points = np.array(all_points)

        # Find only the TOP corners (min y values)
        min_x, min_y = np.min(all_points, axis=0)
        max_x, max_y = np.max(all_points, axis=0)

        # Find closest actual points to top corners only
        top_left_target = [min_x, min_y]
        top_right_target = [max_x, min_y]

        # Find closest actual points to each top corner
        distances_tl = np.sum((all_points - top_left_target) ** 2, axis=1)
        closest_tl_idx = np.argmin(distances_tl)

        distances_tr = np.sum((all_points - top_right_target) ** 2, axis=1)
        closest_tr_idx = np.argmin(distances_tr)

        self.corners = {
            "top_left": tuple(all_points[closest_tl_idx]),
            "top_right": tuple(all_points[closest_tr_idx]),
        }

        self.ref_points = None
        self.gps_points = None

    def show_corners(self):
        """Display the image with highlighted TOP corners only"""
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        # Colors for features
        colors = {
            "bunker": "#8B4513",
            "green": "#228B22",
            "fairway": "#90EE90",
            "rough": "#556B2F",
            "tee": "#FFD700",
        }

        # Draw all detections
        for detection in self.data["detections"]:
            feature_class = detection["class"]
            outline_points = detection["outline_points"]

            polygon = Polygon(
                outline_points,
                facecolor=colors.get(feature_class, "#808080"),
                edgecolor="black",
                alpha=0.7,
                linewidth=1,
            )
            ax.add_patch(polygon)

        # Highlight only the TOP 2 corners
        corner_colors = ["red", "blue"]
        corner_labels = ["TOP-LEFT", "TOP-RIGHT"]

        print("DETECTED TOP CORNERS:")
        for i, (name, (x, y)) in enumerate(self.corners.items()):
            # Draw corner circle
            circle = plt.Circle(
                (x, y),
                radius=25,
                color=corner_colors[i],
                fill=True,
                alpha=0.8,
                zorder=10,
            )
            ax.add_patch(circle)

            # Add label
            ax.text(
                x,
                y - 50,
                f"{corner_labels[i]}\n({x}, {y})",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=corner_colors[i],
                    linewidth=2,
                ),
            )

            print(f"  {name}: ({x}, {y})")

        # Set plot limits
        all_points = [
            p
            for detection in self.data["detections"]
            for p in detection["outline_points"]
        ]
        if all_points:
            xs, ys = zip(*all_points)
            ax.set_xlim(min(xs) - 50, max(xs) + 50)
            ax.set_ylim(min(ys) - 50, max(ys) + 50)

        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title("TOP corners for GPS alignment", fontsize=16, fontweight="bold")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def set_reference_points(
        self, corner1_name, gps1_lat, gps1_lon, corner2_name, gps2_lat, gps2_lon
    ):
        """Set the two reference points with their GPS coordinates"""

        self.ref_points = [self.corners[corner1_name], self.corners[corner2_name]]

        self.gps_points = [(gps1_lat, gps1_lon), (gps2_lat, gps2_lon)]

        # Calculate transformation
        p1_pixel, p2_pixel = self.ref_points
        p1_gps, p2_gps = self.gps_points

        # Calculate differences
        dx_pixel = p2_pixel[0] - p1_pixel[0]
        dy_pixel = p2_pixel[1] - p1_pixel[1]
        dx_gps = p2_gps[1] - p1_gps[1]  # longitude
        dy_gps = p2_gps[0] - p1_gps[0]  # latitude

        # Calculate scale and rotation
        pixel_distance = np.sqrt(dx_pixel**2 + dy_pixel**2)
        gps_distance = np.sqrt(dx_gps**2 + dy_gps**2)

        self.scale = gps_distance / pixel_distance if pixel_distance != 0 else 1

        angle_pixel = np.arctan2(dy_pixel, dx_pixel)
        angle_gps = np.arctan2(dy_gps, dx_gps)
        self.rotation = angle_gps - angle_pixel

        # Calculate translation
        cos_r = np.cos(self.rotation)
        sin_r = np.sin(self.rotation)

        self.translation = (
            p1_gps[0]
            - (p1_pixel[0] * cos_r * self.scale - p1_pixel[1] * sin_r * self.scale),
            p1_gps[1]
            - (p1_pixel[0] * sin_r * self.scale + p1_pixel[1] * cos_r * self.scale),
        )

        print(f"✅ Transformation set using {corner1_name} and {corner2_name}")

    def transform_point(self, pixel_x, pixel_y):
        """Transform a pixel coordinate to GPS"""
        if self.ref_points is None:
            raise ValueError("Set reference points first!")

        cos_r = np.cos(self.rotation)
        sin_r = np.sin(self.rotation)

        lat = (
            pixel_x * cos_r * self.scale
            - pixel_y * sin_r * self.scale
            + self.translation[0]
        )
        lon = (
            pixel_x * sin_r * self.scale
            + pixel_y * cos_r * self.scale
            + self.translation[1]
        )

        return (lat, lon)

    def transform_all(self):
        """Transform all detections to new coordinate system (scaled and rotated but not GPS)"""
        if self.ref_points is None:
            raise ValueError("Set reference points first!")

        for detection in self.data["detections"]:
            transformed_outline = []
            for x, y in detection["outline_points"]:
                # Apply only rotation and scaling, not GPS conversion
                cos_r = np.cos(self.rotation)
                sin_r = np.sin(self.rotation)

                # Transform but keep reasonable scale for fields2cover
                new_x = x * cos_r - y * sin_r
                new_y = x * sin_r + y * cos_r

                transformed_outline.append([new_x, new_y])

            detection["transformed_outline"] = transformed_outline

        return self.data

    def show_transformed_result(self):
        """Show before and after transformation side by side"""
        if self.ref_points is None:
            raise ValueError("Set reference points first!")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Colors for features
        colors = {
            "bunker": "#8B4513",
            "green": "#228B22",
            "fairway": "#90EE90",
            "rough": "#556B2F",
            "tee": "#FFD700",
        }

        # LEFT: Original pixel coordinates
        for detection in self.data["detections"]:
            feature_class = detection["class"]
            outline_points = detection["outline_points"]

            polygon = Polygon(
                outline_points,
                facecolor=colors.get(feature_class, "#808080"),
                edgecolor="black",
                alpha=0.7,
                linewidth=1,
            )
            ax1.add_patch(polygon)

        # Highlight reference points on original
        ref_colors = ["red", "blue"]
        for i, (x, y) in enumerate(self.ref_points):
            circle = plt.Circle(
                (x, y), radius=25, color=ref_colors[i], fill=True, alpha=0.8, zorder=10
            )
            ax1.add_patch(circle)
            ax1.text(
                x,
                y - 50,
                f"REF {i + 1}\n({x}, {y})",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=ref_colors[i],
                    linewidth=2,
                ),
            )

        # Set limits for original
        all_points = [
            p
            for detection in self.data["detections"]
            for p in detection["outline_points"]
        ]
        if all_points:
            xs, ys = zip(*all_points)
            ax1.set_xlim(min(xs) - 50, max(xs) + 50)
            ax1.set_ylim(min(ys) - 50, max(ys) + 50)

        ax1.set_aspect("equal")
        ax1.invert_yaxis()
        ax1.set_title("BEFORE: Pixel Coordinates", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Pixel X")
        ax1.set_ylabel("Pixel Y")
        ax1.grid(True, alpha=0.3)

        # RIGHT: Transformed coordinates (rotated/scaled)
        for detection in self.data["detections"]:
            if "transformed_outline" not in detection:
                continue

            feature_class = detection["class"]
            transformed_outline = detection["transformed_outline"]

            polygon = Polygon(
                transformed_outline,
                facecolor=colors.get(feature_class, "#808080"),
                edgecolor="black",
                alpha=0.7,
                linewidth=1,
            )
            ax2.add_patch(polygon)

        # Highlight transformed reference points
        for i, (pixel_x, pixel_y) in enumerate(self.ref_points):
            # Apply same transformation as shapes
            cos_r = np.cos(self.rotation)
            sin_r = np.sin(self.rotation)

            new_x = pixel_x * cos_r - pixel_y * sin_r
            new_y = pixel_x * sin_r + pixel_y * cos_r

            circle = plt.Circle(
                (new_x, new_y),
                radius=25,
                color=ref_colors[i],
                fill=True,
                alpha=0.8,
                zorder=10,
            )
            ax2.add_patch(circle)

            # Show transformation info
            # ax2.text(
            #     new_x,
            #     new_y - 50,
            #     f"REF {i + 1}\nOriginal: ({pixel_x}, {pixel_y})\nTransformed: ({new_x:.1f}, {new_y:.1f})",
            #     ha="center",
            #     va="center",
            #     fontsize=10,
            #     fontweight="bold",
            #     bbox=dict(
            #         boxstyle="round,pad=0.3",
            #         facecolor="white",
            #         edgecolor=ref_colors[i],
            #         linewidth=2,
            #     ),
            # )

        # Set limits for transformed coordinates
        all_transformed_points = []
        for detection in self.data["detections"]:
            if "transformed_outline" in detection:
                all_transformed_points.extend(detection["transformed_outline"])

        if all_transformed_points:
            xs, ys = zip(*all_transformed_points)
            margin_x = (max(xs) - min(xs)) * 0.1
            margin_y = (max(ys) - min(ys)) * 0.1
            ax2.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
            ax2.set_ylim(min(ys) - margin_y, max(ys) + margin_y)

        ax2.set_aspect("equal")
        ax2.set_title(
            "AFTER: Rotated/Scaled Coordinates", fontsize=14, fontweight="bold"
        )
        ax2.set_xlabel("Transformed X")
        ax2.set_ylabel("Transformed Y")
        ax2.invert_xaxis()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print transformation summary
        print(f"\n📊 TRANSFORMATION SUMMARY:")
        print(f"   Reference Point 1: {self.ref_points[0]} → {self.gps_points[0]}")
        print(f"   Reference Point 2: {self.ref_points[1]} → {self.gps_points[1]}")
        print(f"   Scale Factor: {self.scale:.8f}")
        print(f"   Rotation: {np.degrees(self.rotation):.2f}°")
        print(
            f"   Features Transformed: {len([d for d in self.data['detections'] if 'transformed_outline' in d])}"
        )

    def save_transformed_data(self, output_path=None):
        """Save the transformed data to a new JSON file in exact same format as input"""
        if self.ref_points is None:
            raise ValueError("Set reference points first!")

        if output_path is None:
            # Auto-generate filename
            original_name = self.data["metadata"]["image_name"]
            base_name = original_name.split(".")[0]
            output_path = f"{base_name}_transformed.json"

        # Create new data structure in exact same format
        transformed_data = {"metadata": self.data["metadata"].copy(), "detections": []}

        # Replace outline_points with transformed coordinates
        for detection in self.data["detections"]:
            if "transformed_outline" in detection:
                new_detection = {
                    "class": detection["class"],
                    "confidence": detection["confidence"],
                    "outline_points": detection["transformed_outline"],
                }
                transformed_data["detections"].append(new_detection)

        # Save to file
        with open(output_path, "w") as f:
            json.dump(transformed_data, f, indent=2)

        print(f"💾 Transformed data saved to: {output_path}")
        print(f"   Format: Same as input but with rotated/scaled coordinates")
        return output_path


# Simple usage example
def main():
    # Load and process
    transformer = SimpleGPSTransformer(
        "../outputs/outlines/Benniksgaard_Golf_Klub_1000_02_1_outlines.json"
    )

    # Show corners
    transformer.show_corners()

    # Set reference points (example)
    transformer.set_reference_points(
        "top_left",
        53.889408,
        9.547150,  # GPS for top-left corner
        "top_right",
        54.886907,
        9.546503,  # GPS for top-right corner
    )

    # Transform everything
    transformed_data = transformer.transform_all()

    # Show before and after
    transformer.show_transformed_result()

    # Save the transformed data
    output_file = transformer.save_transformed_data(
        output_path="../outputs/transformedOutlines/Benniksgaard_Golf_Klub_1000_02_1_outlines_transformed.json"
    )

    # Test a point
    lat, lon = transformer.transform_point(1000, 500)
    print(f"Point (1000, 500) → GPS: {lat:.6f}, {lon:.6f}")


if __name__ == "__main__":
    main()

"""

        "top_left",
        53.889408,
        9.547150,  # GPS for top-left corner
        "top_right",
        54.886907,
        9.546503,  # GPS for bottom-right corner




"""
