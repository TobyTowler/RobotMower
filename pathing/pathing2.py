import fields2cover as f2c
import math
from Pathing.mapOrientation import rotateMap, scaleMap
from utils import load_csv_points, genField, drawCell, save_points_to_csv, mowerConfig


def test():
    mower = mowerConfig(2.0, 5.0)

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
    print(cell[0].area())
    # print(path_dubins_cc)


def rotateTest():
    points = load_csv_points("./fields/rotatedField.csv")
    field = genField(points)

    mower = mowerConfig(0.5, 0.3)

    field, swaths, no_hl, path_dubins = genPath(field, mower)

    drawCell([field, swaths, no_hl, path_dubins])
    return [field, swaths, no_hl, path_dubins]


def rotateAndScale():
    points = load_csv_points("./fields/rotatedField.csv")
    field = genField(points)

    mower = mowerConfig(0.5, 0.3)

    rotated = ro

    field, swaths, no_hl, path_dubins = genPath(field, mower)

    drawCell([field, swaths, no_hl, path_dubins])
    return [field, swaths, no_hl, path_dubins]


def main():
    rotateTest()


if __name__ == "__main__":
    main()
