import fields2cover as f2c
import csv


def genField(csv_points):
    # Load the CSV data

    ring = f2c.LinearRing()
    for p in csv_points:
        print(p)
        ring.addGeometry(f2c.Point(p[0], p[1]))

    cell = f2c.Cell()
    cell.addRing(ring)

    cells = f2c.Cells()
    cells.addGeometry(cell)

    return cells


def load_csv_points(path):
    points = []
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile)  # noqa: F821
        # Skip the header line
        next(reader)
        for row in reader:
            # Extract coordinates from columns 1 and 2 (x and y)
            point_id, x, y = int(row[0]), float(row[1]), float(row[2])
            points.append((x, y))
    return points


def mowerConfig(width, bladeWidth):
    """All lengths in m"""

    mower = f2c.Robot(width, bladeWidth)
    mower.setMinTurningRadius(0.4)  # m
    mower.setMaxDiffCurv(0.1)  # 1/m^2
    return mower


def save_points_to_csv(points, filename):
    """
    Saves a list of points to a CSV file in the format point_id,x,y

    Args:
        points (list): List of points as (x, y) tuples
        filename (str): Name of the file to save the points to
    """
    with open("fields/" + filename + ".csv", "w") as file:
        # Write header
        file.write("point_id,x,y\n")

        # Write points with IDs
        for i, point in enumerate(points, 1):  # Start IDs from 1
            x, y = point
            file.write(f"{i},{x},{y}\n")

    print(f"Points successfully saved to {filename}")


def drawCell(arr):
    f2c.Visualizer.figure()
    for i in arr:
        f2c.Visualizer.plot(i)
    f2c.Visualizer.show()
