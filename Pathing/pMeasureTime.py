import time
import matplotlib.pyplot as plt
import statistics
import fields2cover as f2c
import math
from utils import mowerConfig, load_csv_points, genField, drawCell


def main():
    mower = mowerConfig(0.22, 0.15)

    rand = f2c.Random(42)
    field = rand.generateRandField(1e4, 6)
    # hole = rand.generateRandCell(121, 4)
    # # hole1 = hole.getField()
    cell = field.getField()
    print(cell)
    # cell = f2c.Cell()

    # cell = fieldConfig21313(6)

    const_hl = f2c.HG_Const_gen()
    no_hl = const_hl.generateHeadlands(cell, 3.0 * mower.getWidth())

    bf = f2c.SG_BruteForce()
    swaths = bf.generateSwaths(math.pi, mower.getCovWidth(), no_hl.getGeometry(0))
    snake_sorter = f2c.RP_Snake()
    swaths = snake_sorter.genSortedSwaths(swaths)

    path_planner = f2c.PP_PathPlanning()
    dubins_cc = f2c.PP_DubinsCurvesCC()
    path_dubins_cc = path_planner.planPath(mower, swaths, dubins_cc)

    drawCell([cell, swaths, no_hl, path_dubins_cc])


if __name__ == "__main__":
    main()
