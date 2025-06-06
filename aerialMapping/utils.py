import torch
import torchvision
import numpy as np
import cv2
import json
import os
from datetime import datetime
from PIL import Image
from aerialMapping.maskRCNNmodel import get_model


def save_outlines_to_json(outlines, image_path):
    output_dir = "outputs/outlines"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    json_data = {
        "metadata": {
            "image_path": image_path,
            "image_name": os.path.basename(image_path),
            "timestamp": datetime.now().isoformat(),
            "total_detections": len(outlines),
        },
        "detections": outlines,
    }

    json_path = os.path.join(output_dir, f"{base_name}_outlines.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Outlines saved to: {json_path}")
    return json_path


def save_route_to_json(route, filename):
    if route is None:
        print("No route to save")
        return

    print("Route type:", type(route))
    print("Route methods:", [m for m in dir(route) if not m.startswith("_")])

    waypoints = []

    try:
        if hasattr(route, "size"):
            size = route.size()
            print(f"Route size: {size}")

            for i in range(size):
                try:
                    point = route[i]
                    waypoints.append({"index": i, "x": point.getX(), "y": point.getY()})
                except:
                    print(f"Failed to access route[{i}]")
                    break

        elif hasattr(route, "getPath"):
            path = route.getPath()
            print(f"Path type: {type(path)}")

        elif hasattr(route, "getGeometry"):
            geom = route.getGeometry()

        else:
            print("Unknown route structure")

    except Exception as e:
        print(f"Error extracting route data: {e}")
        return

    if waypoints:
        route_data = {
            "metadata": {
                "total_waypoints": len(waypoints),
                "route_length": route.length() if hasattr(route, "length") else 0,
            },
            "waypoints": waypoints,
        }

        with open(filename, "w") as f:
            json.dump(route_data, f, indent=2)

        print(f"Route saved to {filename} with {len(waypoints)} waypoints")
    else:
        print("No waypoints extracted from route")
