import math
from pathing.utils import load_csv_points, genField, drawCell, save_points_to_csv

import geopy.distance
import json
from datetime import datetime

"""
get shape
get gps

sheer shape to gps
rotate shape
scale shape
transform to gps coordinates

virtual map
scale to size
rotate to angle

"""


def getAngle(gps):
    if len(gps) < 2:
        raise ValueError("Need at least 2 GPS coordinates to calculate bearing")

    lat1, lon1 = gps[0]
    lat2, lon2 = gps[1]

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # θ = atan2(sin(Δlong) * cos(lat2), cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(Δlong))
    delta_lon = lon2_rad - lon1_rad
    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(
        lat2_rad
    ) * math.cos(delta_lon)

    return math.atan2(x, y)


def rotateMap(points, gps):
    rotation_angle = -(getAngle(gps))

    cos_angle = math.cos(rotation_angle)
    sin_angle = math.sin(rotation_angle)

    rotated_points = []
    for point in points:
        x, y = point

        rotated_x = x * cos_angle - y * sin_angle
        rotated_y = x * sin_angle + y * cos_angle
        rotated_points.append((rotated_x, rotated_y))

    return rotated_points


def sheerMap(map, gps):
    gpsMeters = geopy.distance.geodesic(gps[0], gps[1]).m

    gpsAngle = getAngle(gps)
    gpsMapCoords = [map[0], map[1]]
    mapAngle = getAngle(gpsMapCoords)

    if gpsAngle == mapAngle:
        return map


def translateMap(map):
    pass


def getLineLength(map: list[float]):
    print(map)
    x1, y1, x2, y2 = map
    x = abs(x1 - x2)
    y = abs(y1 - y2)
    return math.sqrt(x**2 + y**2)


def scaleMap(map, gps):
    gpsMeters = geopy.distance.geodesic(gps[0], gps[1]).m

    mapLineLength = getLineLength([map[0][0], map[0][1], map[1][0], map[1][1]])

    print("GPS ", gpsMeters, "MAP", mapLineLength)
    scale = gpsMeters / mapLineLength
    print("SCALE", scale)
    scaledMap = [map[0]]

    for p in range(1, len(map)):
        start = map[p - 1]
        point = map[p]
        newPoint = scaledMap[-1]

        print("START ", start, "\nPOINT ", point)

        if start[0] == point[0]:
            x = newPoint[0]
        elif start[0] > point[0]:
            x = newPoint[0] - (abs(start[0] - point[0]) * scale)
        else:
            x = newPoint[0] + (abs(start[0] - point[0]) * scale)

        if start[1] == point[1]:
            y = newPoint[1]
        elif start[1] > point[1]:
            y = newPoint[1] - (abs(start[1] - point[1]) * scale)
        else:
            y = newPoint[1] + (abs(start[1] - point[1]) * scale)
        scaledMap.append([x, y])

    return scaledMap


def save_points_to_json(points, filename, point_class="fairway", confidence=0.95):
    outline_points = []
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            outline_points.append([int(point[0]), int(point[1])])

    detection_data = {
        "metadata": {
            "image_path": f"/path/to/{filename}.jpg",
            "image_name": f"{filename}.jpg",
            "timestamp": datetime.now().isoformat(),
            "total_detections": 1,
        },
        "detections": [
            {
                "class": point_class,
                "confidence": confidence,
                "outline_points": outline_points,
            }
        ],
    }

    output_filename = f"{filename}.json"
    with open(output_filename, "w") as f:
        json.dump(detection_data, f, indent=2)

    print(f"Points saved to {output_filename} in JSON format")
    return output_filename


def transformPoints(json_path, gps_coords):
    try:
        with open(json_path, "r") as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {json_path}: {e}")
        return None

    if len(gps_coords) < 2:
        print("Error: Need at least 2 GPS coordinates for transformation")

    print(f"Transforming detections using GPS coordinates: {gps_coords}")

    transformed_detections = []
    for i, detection in enumerate(json_data["detections"]):
        print(f"Transforming detection {i + 1}: {detection['class']}")

        original_points = detection["outline_points"]

        try:
            rotated_points = rotateMap(original_points, gps_coords)

            scaled_points = scaleMap(rotated_points, gps_coords)

            transformed_detection = {
                "class": detection["class"],
                "confidence": detection["confidence"],
                "outline_points": [
                    [int(point[0]), int(point[1])] for point in scaled_points
                ],
            }

            transformed_detections.append(transformed_detection)
            print(f"  Transformed {len(original_points)} points")

        except Exception as e:
            print(f"Error transforming detection {i + 1}: {e}")
            transformed_detections.append(detection)

    transformed_json_data = {
        "metadata": {
            "image_path": json_data["metadata"]["image_path"],
            "image_name": json_data["metadata"]["image_name"],
            "timestamp": datetime.now().isoformat(),
            "total_detections": len(transformed_detections),
            "gps_transformed": True,
            "gps_reference_points": gps_coords,
            "original_detections": json_data["metadata"]["total_detections"],
        },
        "detections": transformed_detections,
    }

    base_path = json_path.replace(".json", "")
    output_path = "outputs/transformedOutlines/"

    try:
        with open(output_path, "w") as f:
            json.dump(transformed_json_data, f, indent=2)

        print(f"GPS-transformed detections saved to: {output_path}")
        print(f"Transformed {len(transformed_detections)} detections")

        return output_path

    except Exception as e:
        print(f"Error saving transformed JSON: {e}")


def main():
    path = "coords/garden.csv"
    gpsPath = "gps/garden.csv"
    # path = "coords/AdobeGold_golf_course_outline.csv"
    gps = load_csv_points(gpsPath, reverse=True)
    points = load_csv_points(path)

    field = genField(points)
    gpsField = genField(gps)
    drawCell(field)
    # drawCell(gpsField)

    newPoints = rotateMap(points, gps)

    # drawCell(genField(newPoints))

    scaledPoints = scaleMap(points, gps)
    scaledField = genField(scaledPoints)
    drawCell(scaledField)

    save_points_to_csv(newPoints, "rotatedField")


if __name__ == "__main__":
    main()
