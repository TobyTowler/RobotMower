import fields2cover as f2c
import numpy as np
import matplotlib.pyplot as plt
import csv
import math


def drawCell(arr):
    f2c.Visualizer.figure()
    for i in arr:
        f2c.Visualizer.plot(i)
    f2c.Visualizer.show()


def mowerConfig(length, width):
    """The vehicle to cover the field is defined as a F2CRobot struct. To initialize it, the constructor needs the width of the robot and the width of the operation. For example, if we have a vehicle to fertilize a field, with 3m width and a 39m operational width, we should initialize it as:"""
    mower = f2c.Robot(length, width)
    mower.setMinTurningRadius(2)  # m
    mower.setMaxDiffCurv(0.1)  # 1/m^2
    return mower


def load_csv_points(path):
    points = []
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile)
        # Skip the header line
        next(reader)
        for row in reader:
            # Extract coordinates from columns 1 and 2 (x and y)
            point_id, x, y = int(row[0]), float(row[1]), float(row[2])
            points.append((x, y))
    return points


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


def main():
    mower = mowerConfig(3.0, 3.0)

    path = "coords/AdobeGold_golf_course_outline.csv"
    points = load_csv_points(path)
    # Generate the field
    cell = genField(points)

    drawCell(cell)

    print(f"Number of cells in field: {cell.size()}")

    # Visualize the field
    # fig, ax = plt.subplots(figsize=(10, 10))

    # Extract coordinates for plotting
    # xs = [p[0] for p in points]
    # ys = [p[1] for p in points]
    #
    # # Plot the field boundary
    # ax.plot(xs, ys, "b-", linewidth=2)
    # ax.fill(xs, ys, alpha=0.3, color="green")
    # ax.set_aspect("equal")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_title("AdobeGold Field")
    # ax.grid(True)

    # plt.show()

    const_hl = f2c.HG_Const_gen()
    no_hl = const_hl.generateHeadlands(cell, 3.0 * mower.getWidth())

    bf = f2c.SG_BruteForce()
    swaths = bf.generateSwaths(math.pi, mower.getCovWidth(), no_hl.getGeometry(0))
    snake_sorter = f2c.RP_Snake()
    swaths = snake_sorter.genSortedSwaths(swaths)

    path_planner = f2c.PP_PathPlanning()
    dubins_cc = f2c.PP_DubinsCurvesCC()
    path_dubins_cc = path_planner.planPath(mower, swaths, dubins_cc)

    # drawCell([cell, swaths, no_hl, path_dubins_cc])
    # drawCell([cell, swaths, no_hl])
    # drawCell(no_hl)
    print(cell[0].area())


if __name__ == "__main__":
    main()
