import fields2cover as f2c
import math
from utils import mowerConfig, load_csv_points, genField, drawCell


def main():
    mower = mowerConfig(3.0, 3.0)

    path = "coords/AdobeGold_golf_course_outline.csv"
    points = load_csv_points(path)
    # Generate the field
    cell = genField(points)

    drawCell(cell)

    print(f"Number of cells in field: {cell.size()}")

    const_hl = f2c.HG_Const_gen()
    no_hl = const_hl.generateHeadlands(cell, 3.0 * mower.getWidth())

    bf = f2c.SG_BruteForce()
    swaths = bf.generateSwaths(math.pi, mower.getCovWidth(), no_hl.getGeometry(0))
    # snake_sorter = f2c.RP_Snake()
    # swaths = snake_sorter.genSortedSwaths(swaths)
    boustrophedon_sorter = f2c.RP_Boustrophedon()
    swaths = boustrophedon_sorter.genSortedSwaths(swaths)
    path_planner = f2c.PP_PathPlanning()
    dubins_cc = f2c.PP_DubinsCurvesCC()
    path_dubins_cc = path_planner.planPath(mower, swaths, dubins_cc)

    # drawCell([cell, swaths, no_hl, path_dubins_cc])
    # drawCell([cell, swaths, no_hl])
    # drawCell(no_hl)
    print(cell[0].area())


if __name__ == "__main__":
    main()
