import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def transformPoints(json_file_path, gps_coords):
    """
    Transform points using GPS coordinates
    gps_coords should be a list of two GPS coordinate strings like:
    ["53.889408,9.547150", "54.886907,9.546503"]
    """
    # Parse GPS coordinates
    try:
        gps1_parts = gps_coords[0].split(",")
        gps2_parts = gps_coords[1].split(",")

        gps1_lat, gps1_lon = float(gps1_parts[0]), float(gps1_parts[1])
        gps2_lat, gps2_lon = float(gps2_parts[0]), float(gps2_parts[1])

        print(f"GPS Coord 1: {gps1_lat}, {gps1_lon}")
        print(f"GPS Coord 2: {gps2_lat}, {gps2_lon}")

    except (ValueError, IndexError) as e:
        print(f"Error parsing GPS coordinates: {e}")
        return json_file_path  # Return original path if GPS parsing fails

    # Initialize transformer
    transformer = SimpleGPSTransformer(json_file_path)

    # Set reference points using top corners
    transformer.set_reference_points(
        "top_left", gps1_lat, gps1_lon, "top_right", gps2_lat, gps2_lon
    )

    # Transform all detections
    transformer.transform_all()

    # Save transformed data
    output_path = json_file_path.replace(".json", "_transformed.json")
    transformer.save_transformed_data(output_path)

    print(f"Transformed coordinates saved to: {output_path}")
    return output_path


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

        print(f"✅ Transformation set using {corner1_name} and {corner2_name}")
        print(f"   Scale: {self.scale:.8f}, Rotation: {np.degrees(self.rotation):.2f}°")

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
        return output_path
