import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
from aerialMapping.maskRCNNmodel import get_model


def run_model_and_get_outlines(image_path):
    class_names = ["background", "green", "fairway", "bunker", "rough", "water"]

    model_path = "aerialMapping/models/golf_course_model_best.pth"
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
        if score >= 0.7:
            mask_np = mask[0].cpu().numpy() > 0.5
            mask_uint8 = (mask_np * 255).astype(np.uint8)

            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)

                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
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

    return results


def main():
    image_path = "./imgs/rawImgs/Benniksgaard_Golf_Klub_1000_02_2.jpg"

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

        return outlines

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    main()
