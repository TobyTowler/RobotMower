import json
import fields2cover as f2c
from shapely.geometry import Polygon, Point
import numpy as np


def load_golf_course_data(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def points_to_polygon(points):
    if len(points) < 3:
        return None

    if points[0] != points[-1]:
        points.append(points[0])
    return Polygon(points)


def points_to_f2c_linestring(points):
    line = f2c.LineString()
    for point in points:
        line.addPoint(f2c.Point(float(point[0]), float(point[1])))
    return line


def points_to_f2c_polygon(points):
    if len(points) < 3:
        return None

    if points[0] != points[-1]:
        points.append(points[0])

    line = points_to_f2c_linestring(points)
    ring = f2c.LinearRing(line)
    return f2c.Polygon(ring)


def find_bunkers_in_fairway(fairway_polygon, bunkers):
    fairway_shape = points_to_polygon(fairway_polygon["outline_points"])
    if fairway_shape is None:
        return []

    bunkers_inside = []
    for bunker in bunkers:
        bunker_shape = points_to_polygon(bunker["outline_points"])
        if bunker_shape is None:
            continue

        bunker_center = bunker_shape.centroid
        if fairway_shape.contains(bunker_center):
            bunkers_inside.append(bunker)

    return bunkers_inside


def create_field_with_holes(fairway, bunkers_inside):
    main_polygon = points_to_f2c_polygon(fairway["outline_points"])
    if main_polygon is None:
        return None

    field = f2c.Field()
    field.setField(main_polygon)

    for bunker in bunkers_inside:
        hole_polygon = points_to_f2c_polygon(bunker["outline_points"])
        if hole_polygon is not None:
            field.addHole(hole_polygon)

    return field


def generate_coverage_paths(field, implement_width=2.0, headland_width=4.0):
    if field is None:
        return None

    robot = f2c.Robot(implement_width, 0.5)

    headlands = f2c.HG_Const_gen()
    no_hl = headlands.generateHeadlands(field, headland_width)

    remaining_area = field.getField()

    decomp = f2c.SG_BruteForce()
    swaths = decomp.generateBestSwaths(
        f2c.OBJ_NSwath(), remaining_area, implement_width
    )

    path_planner = f2c.PP_PathPlanning()
    path = path_planner.searchBestPath(robot, swaths, f2c.OBJ_PathLength())

    return {"headlands": no_hl, "swaths": swaths, "path": path}


def process_golf_course_paths(json_path, implement_width=2.0, headland_width=4.0):
    data = load_golf_course_data(json_path)
    detections = data["detections"]

    fairways = [d for d in detections if d["class"] == "fairway"]
    bunkers = [d for d in detections if d["class"] == "bunker"]
    greens = [d for d in detections if d["class"] == "green"]

    results = {"fairways": [], "greens": [], "standalone_areas": []}

    print(
        f"Processing {len(fairways)} fairways, {len(bunkers)} bunkers, {len(greens)} greens"
    )

    for i, fairway in enumerate(fairways):
        print(f"Processing fairway {i + 1}...")

        bunkers_inside = find_bunkers_in_fairway(fairway, bunkers)
        print(f"  Found {len(bunkers_inside)} bunkers inside fairway")

        field = create_field_with_holes(fairway, bunkers_inside)

        if field is not None:
            paths = generate_coverage_paths(field, implement_width, headland_width)

            results["fairways"].append(
                {
                    "fairway_id": i,
                    "confidence": fairway["confidence"],
                    "bunkers_count": len(bunkers_inside),
                    "field": field,
                    "paths": paths,
                }
            )

    for i, green in enumerate(greens):
        print(f"Processing green {i + 1}...")

        green_polygon = points_to_f2c_polygon(green["outline_points"])
        if green_polygon is not None:
            field = f2c.Field()
            field.setField(green_polygon)

            paths = generate_coverage_paths(
                field, implement_width=1.0, headland_width=2.0
            )

            results["greens"].append(
                {
                    "green_id": i,
                    "confidence": green["confidence"],
                    "field": field,
                    "paths": paths,
                }
            )

    return results


def save_paths_as_coordinates(results, output_path="golf_paths.json"):
    def f2c_path_to_coordinates(path):
        """Convert Fields2Cover path to coordinate list"""
        coords = []
        if path is None:
            return coords

        try:
            for i in range(path.size()):
                swath = path.getSwath(i)
                for j in range(swath.getPath().size()):
                    point = swath.getPath().getGeometry(j)
                    coords.append([point.getX(), point.getY()])
        except:
            print("Warning: Could not extract path coordinates")

        return coords

    output = {
        "metadata": {
            "total_fairways": len(results["fairways"]),
            "total_greens": len(results["greens"]),
        },
        "fairway_paths": [],
        "green_paths": [],
    }

    for fairway_result in results["fairways"]:
        if fairway_result["paths"] and fairway_result["paths"]["path"]:
            path_coords = f2c_path_to_coordinates(fairway_result["paths"]["path"])

            output["fairway_paths"].append(
                {
                    "fairway_id": fairway_result["fairway_id"],
                    "bunkers_avoided": fairway_result["bunkers_count"],
                    "path_coordinates": path_coords,
                }
            )

    for green_result in results["greens"]:
        if green_result["paths"] and green_result["paths"]["path"]:
            path_coords = f2c_path_to_coordinates(green_result["paths"]["path"])

            output["green_paths"].append(
                {"green_id": green_result["green_id"], "path_coordinates": path_coords}
            )

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Paths saved to {output_path}")
    return output_path


def main():
    json_path = "Benniksgaard_Golf_Klub_1000_02_2_outlines.json"

    try:
        results = process_golf_course_paths(json_path, implement_width=2.0)

        output_path = save_paths_as_coordinates(results)

        print(
            f"Generated paths for {len(results['fairways'])} fairways and {len(results['greens'])} greens"
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
