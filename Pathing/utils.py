import fields2cover as f2c
import csv


def genField(csv_points):
    ring = f2c.LinearRing()
    for p in csv_points:
        print(p)
        ring.addGeometry(f2c.Point(p[0], p[1]))

    cell = f2c.Cell()
    cell.addRing(ring)

    cells = f2c.Cells()
    cells.addGeometry(cell)

    return cells


# Possibly need to change reverse to be in drawCell
def load_csv_points(path, reverse=False):
    points = []
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile)

        # Skip the header line
        next(reader)
        for row in reader:
            if reverse:
                point_id, y, x = int(row[0]), float(row[1]), float(row[2])
            else:
                point_id, x, y = int(row[0]), float(row[1]), float(row[2])
            points.append((x, y))
    return points


def mowerConfig(width, bladeWidth):
    """All lengths in m"""

    mower = f2c.Robot(width, bladeWidth)
    mower.setMinTurningRadius(0.15)  # m
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
        file.write("point_id,x,y\n")

        for i, point in enumerate(points, 1):  # Start IDs from 1
            x, y = point
            file.write(f"{i},{x},{y}\n")

    print(f"Points successfully saved to {filename}")


def drawCell(arr):
    f2c.Visualizer.figure()
    for i in arr:
        f2c.Visualizer.plot(i)
    f2c.Visualizer.show()


def getRobotCoords():
    return [2, 4]


def genPath(field, mower):
    const_hl = f2c.HG_Const_gen()
    no_hl = const_hl.generateHeadlands(field, 3.0 * mower.getWidth())

    # bf = f2c.SG_BruteForce()

    n_swath = f2c.OBJ_NSwath()
    bf_sw_gen = f2c.SG_BruteForce()
    swaths_bf_nswath = bf_sw_gen.generateBestSwaths(
        n_swath, mower.getCovWidth(), no_hl.getGeometry(0)
    )
    # swaths = bf.generateSwaths(math.pi, mower.getCovWidth(), no_hl.getGeometry(0))
    boustrophedon_sorter = f2c.RP_Boustrophedon()
    # swaths = boustrophedon_sorter.genSortedSwaths(swaths)
    swaths = boustrophedon_sorter.genSortedSwaths(swaths_bf_nswath)

    path_planner = f2c.PP_PathPlanning()
    dubins = f2c.PP_DubinsCurves()
    path_dubins = path_planner.planPath(mower, swaths, dubins)

    return [field, swaths, no_hl, path_dubins]
