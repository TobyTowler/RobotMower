import json
import math
import fields2cover as f2c

from utils import drawCell


def create_cells_from_fairway_outlines(json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    fairways = [
        detection for detection in data["detections"] if detection["class"] == "fairway"
    ]

    print(f"Found {len(fairways)} fairway detections")

    cells = []

    for i, fairway in enumerate(fairways):
        outline_points = fairway["outline_points"]
        confidence = fairway["confidence"]

        print(f"\nProcessing Fairway {i + 1}:")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Number of outline points: {len(outline_points)}")

        # Convert outline points to f2c.Point format and add to PointVector
        point_vector = f2c.VectorPoint()
        for point in outline_points:
            x, y = point[0], point[1]
            # Scale pixel coordinates to meters (adjust scale as needed)
            scale = 0.1  # 1 pixel = 0.1 meters
            point_vector.append(f2c.Point(x * scale, y * scale))

        # Close the ring by adding the first point at the end if not already closed
        first_point = f2c.Point(outline_points[0][0] * 0.1, outline_points[0][1] * 0.1)
        last_point = f2c.Point(outline_points[-1][0] * 0.1, outline_points[-1][1] * 0.1)
        if (
            first_point.getX() != last_point.getX()
            or first_point.getY() != last_point.getY()
        ):
            point_vector.append(first_point)

        ring = f2c.LinearRing(point_vector)
        cell = f2c.Cell(ring)
        cells.append(cell)

        print(f"  Created cell with area: {cell.area():.2f} square meters")

    return cells


def pixel_to_world_example(x_pixel, y_pixel):
    """
    Example transformation from pixel coordinates to world coordinates
    """
    scale = 0.1  # meters per pixel
    x_world = x_pixel * scale
    y_world = -y_pixel * scale  # Negative because image Y increases downward
    return x_world, y_world


if __name__ == "__main__":
    json_file = "../outputs/outlines/Benniksgaard_Golf_Klub_1000_02_2_outlines.json"

    cells_list = create_cells_from_fairway_outlines(json_file)
    print(f"\nCreated {len(cells_list)} cells from fairway detections")

    print(f"LEN CELLS = {len(cells_list)}")
    if not cells_list:
        print("No cells created!")
        exit()

    cells = f2c.Cells()
    for cell in cells_list:
        cells.addGeometry(cell)
        print(f"Added cell with area: {cell.area():.2f} square meters")

    # Robot setup - use reasonable size (2 meters working width)
    robot = f2c.Robot(1.0)
    print(f"Robot width: {robot.getWidth():.2f} meters")
    print(f"Robot coverage width: {robot.getCovWidth():.2f} meters")

    # Create field from cells
    field = f2c.Field(cells)
    print(f"Field area: {field.area():.2f} square meters")

    # Generate headlands
    print("Generating headlands...")
    const_hl = f2c.HG_Const_gen()

    try:
        # Headlands for turning (outer)
        mid_hl = const_hl.generateHeadlands(cells, 1.5 * robot.getWidth())
        no_hl = const_hl.generateHeadlands(cells, 3.0 * robot.getWidth())
    except Exception as e:
        print(f"Error generating headlands: {e}")
        print("This might be due to the field being too small or invalid geometry")
        # Use original cells if headland generation fails

    # Generate swaths
    print("Generating swaths...")
    try:
        bf = f2c.SG_BruteForce()
        swaths = bf.generateSwaths(math.pi / 2.0, robot.getCovWidth(), no_hl)
        print(f"Generated {len(swaths)} swaths")
    except Exception as e:
        print(f"Error generating swaths: {e}")

    # Sort swaths (Boustrophedon pattern)
    print("Sorting swaths...")
    try:
        route_planner = f2c.RP_RoutePlannerBase()
        route = route_planner.genRoute(mid_hl, swaths)
        print("Swaths sorted successfully")
    except Exception as e:
        print(f"Error sorting swaths: {e}")
        sorted_swaths = swaths

    # Visualize results
    print("Drawing results...")
    try:
        drawCell([cells, mid_hl, route])
        print("Visualization complete")
    except Exception as e:
        print(f"Error in visualization: {e}")
        print("Check your drawCell function and ensure it can handle the data types")

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Original cells: {len(cells_list)}")
    print(f"Total field area: {field.area():.2f} square meters")
    if len(outer_headlands) > 0:
        total_headland_area = sum(cell.area() for cell in outer_headlands)
        print(f"Headland area: {total_headland_area:.2f} square meters")
    print(f"Number of swaths: {len(swaths)}")
    if len(swaths) > 0:
        total_swath_length = sum(swath.length() for swath in swaths)
        print(f"Total swath length: {total_swath_length:.2f} meters")


def create_single_cell_example():
    """Simple example with just one cell"""

    # Use the first fairway from your data
    fairway_coords = [
        [1598, 58],
        [1274, 88],
        [1124, 162],
        [1087, 246],
        [1256, 273],
        [1573, 168],
    ]

    # Create points with scaling
    points = f2c.VectorPoint()
    scale = 0.1
    for coord in fairway_coords:
        points.append(f2c.Point(coord[0] * scale, coord[1] * scale))

    # Close the ring
    points.append(f2c.Point(fairway_coords[0][0] * scale, fairway_coords[0][1] * scale))

    # Create cell
    ring = f2c.LinearRing(points)
    cell = f2c.Cell(ring)

    # Create cells container
    cells = f2c.Cells()
    cells.addGeometry(cell)

    # Generate headlands
    robot = f2c.Robot(2.0)
    hg = f2c.HG_Const_gen()
    headlands = hg.generateHeadlands(cells, 3.0)

    print(f"Single cell area: {cell.area():.2f} square meters")
    print(f"Generated {len(headlands)} headland cells")

    return cell, headlands


def create_cells_from_fairway_list(fairway_list):
    """Create cells directly from the fairway list"""
    cells = []

    for i, fairway in enumerate(fairway_list):
        point_vector = f2c.VectorPoint()
        scale = 0.1  # Add scaling here too

        for point in fairway["outline_points"]:
            x, y = point[0] * scale, point[1] * scale
            point_vector.append(f2c.Point(x, y))

        # Close the ring
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
            point_vector.append(first_point)

        ring = f2c.LinearRing(point_vector)
        cell = f2c.Cell(ring)
        cells.append(cell)

        print(
            f"Fairway {i + 1}: {len(fairway['outline_points'])} points, "
            f"confidence: {fairway['confidence']:.4f}, "
            f"area: {cell.area():.2f} square meters"
        )

    return cells
