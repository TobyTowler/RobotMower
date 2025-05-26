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

    # Get base filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Create JSON data
    json_data = {
        "metadata": {
            "image_path": image_path,
            "image_name": os.path.basename(image_path),
            "timestamp": datetime.now().isoformat(),
            "total_detections": len(outlines),
        },
        "detections": outlines,
    }

    # Save to file
    json_path = os.path.join(output_dir, f"{base_name}_outlines.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Outlines saved to: {json_path}")
    return json_path
