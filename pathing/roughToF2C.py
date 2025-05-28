import fields2cover as f2c
import json
import math
from shapely.geometry import Polygon, Point

from pathing.utils import drawCell


def create_rough_cells_with_holes(json_file_path, scale=0.1):
    """Create fairway cells with holes using addRing for overlapping obstacles"""
    with open(json_file_path, "r") as f:
        data = json.load(f)

    fairways = [d for d in data["detections"] if d["class"] == "rough"]
    obstacles = [
        d
        for d in data["detections"]
        if d["class"] in ["bunker", "green", "water", "fairway"]
    ]

    cells = f2c.Cells()

    for i, fairway in enumerate(fairways):
        print(f"Processing fairway {i + 1}")

        points = f2c.VectorPoint()
        for point in fairway["outline_points"]:
            points.append(f2c.Point(point[0] * scale, point[1] * scale))

        first_point = f2c.Point(
            fairway["outline_points"][0][0] * scale,
            fairway["outline_points"][0][1] * scale,
        )
        last_point = f2c.Point(
            fairway["outline_points"][-1][0] * scale,
            fairway["outline_points"][-1][1] * scale,
        )
        if (
            first_point.getX() != last_point.getX()
            or first_point.getY() != last_point.getY()
        ):
            points.append(first_point)

        ring = f2c.LinearRing(points)
        cell = f2c.Cell(ring)
        cells.addGeometry(cell)

        fairway_coords = [
            (p[0] * scale, p[1] * scale) for p in fairway["outline_points"]
        ]
        fairway_poly = Polygon(fairway_coords)

        hole_count = 0
        for obstacle in obstacles:
            obstacle_coords = [
                (p[0] * scale, p[1] * scale) for p in obstacle["outline_points"]
            ]
            obstacle_poly = Polygon(obstacle_coords)

            if fairway_poly.intersects(obstacle_poly):
                print(f"  Adding {obstacle['class']} as hole")

                hole_points = f2c.VectorPoint()
                for point in obstacle["outline_points"]:
                    hole_points.append(f2c.Point(point[0] * scale, point[1] * scale))

                first_hole_point = f2c.Point(
                    obstacle["outline_points"][0][0] * scale,
                    obstacle["outline_points"][0][1] * scale,
                )
                last_hole_point = f2c.Point(
                    obstacle["outline_points"][-1][0] * scale,
                    obstacle["outline_points"][-1][1] * scale,
                )
                if (
                    first_hole_point.getX() != last_hole_point.getX()
                    or first_hole_point.getY() != last_hole_point.getY()
                ):
                    hole_points.append(first_hole_point)

                hole_ring = f2c.LinearRing(hole_points)

                cells.addRing(i, hole_ring)
                hole_count += 1

        print(f"  Added {hole_count} holes to fairway {i + 1}")
        print(f"  Fairway area: {cells[i].area():.2f} square meters")

    return cells


def create_obstacle_cells(
    json_file_path, obstacle_classes=["bunker", "green", "water", "fairway"], scale=0.1
):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    obstacles = [d for d in data["detections"] if d["class"] in obstacle_classes]
    obstacle_cells = f2c.Cells()

    for obstacle in obstacles:
        points = f2c.VectorPoint()
        for point in obstacle["outline_points"]:
            points.append(f2c.Point(point[0] * scale, point[1] * scale))

        first_point = f2c.Point(
            obstacle["outline_points"][0][0] * scale,
            obstacle["outline_points"][0][1] * scale,
        )
        last_point = f2c.Point(
            obstacle["outline_points"][-1][0] * scale,
            obstacle["outline_points"][-1][1] * scale,
        )
        if (
            first_point.getX() != last_point.getX()
            or first_point.getY() != last_point.getY()
        ):
            points.append(first_point)

        ring = f2c.LinearRing(points)
        cell = f2c.Cell(ring)
        obstacle_cells.addGeometry(cell)

    return obstacle_cells


def genRoughPath(json_file):
    cells = create_rough_cells_with_holes(json_file, scale=1)
    print(f"Created {len(cells)} fairway cells with holes")

    robot = f2c.Robot(3, 5)
    print(f"Robot width: {robot.getWidth():.2f} meters")
    print(f"Robot coverage width: {robot.getCovWidth():.2f} meters")

    field = f2c.Field(cells)
    print(f"Field area: {field.area():.2f} square meters")

    print("Generating headlands...")
    const_hl = f2c.HG_Const_gen()

    try:
        mid_hl = const_hl.generateHeadlands(cells, 1.5 * robot.getWidth())
        no_hl = const_hl.generateHeadlands(cells, 3.0 * robot.getWidth())
    except Exception as e:
        print(f"Error generating headlands: {e}")
        print("This might be due to the field being too small or invalid geometry")

    print("Generating swaths...")
    try:
        bf = f2c.SG_BruteForce()
        swaths = bf.generateSwaths(math.pi / 2.0, robot.getCovWidth(), no_hl)
        print(f"Generated {len(swaths)} swaths")
    except Exception as e:
        print(f"Error generating swaths: {e}")

    print("Sorting swaths...")
    try:
        route_planner = f2c.RP_RoutePlannerBase()
        route = route_planner.genRoute(mid_hl, swaths)
        print("Swaths sorted successfully")
    except Exception as e:
        print(f"Error sorting swaths: {e}")
        sorted_swaths = swaths

    print("Drawing results...")
    try:
        drawCell([cells, mid_hl, route])
        print("Visualization complete")
    except Exception as e:
        print(f"Error in visualization: {e}")
        print("Check your drawCell function and ensure it can handle the data types")

    return route


if __name__ == "__main__":
    json_file = "../outputs/outlines/Benniksgaard_Golf_Klub_1000_02_2_outlines.json"
    genRoughPath(json_file)
