import fields2cover as f2c
import math
import random


def mowerConfig(length, width):
    mower = f2c.Robot(length, width)
    mower.setMinTurningRadius(2)  # m
    mower.setMaxDiffCurv(0.1)  # 1/m^2
    return mower


def drawCell(arr):
    f2c.Visualizer.figure()
    for i in arr:
        f2c.Visualizer.plot(i)
    f2c.Visualizer.show()
    # f2c.Visualizer.save("Tutorial_image.pn


def main():
    mower = mowerConfig(2.0, 5.0)

    rand = f2c.Random(42)
    field = rand.generateRandField(1e4, 6)
    # hole = rand.generateRandCell(121, 4)
    # # hole1 = hole.getField()
    cell = field.getField()
    # print(cell)
    # cell = f2c.Cell()

    # cell = fieldConfig21313(6)

    const_hl = f2c.HG_Const_gen()
    # no_hl = const_hl.generateHeadlands(cell, 3.0 * mower.getWidth())
    no_hl = const_hl.generateHeadlands(cell, 0)

    bf = f2c.SG_BruteForce()
    swaths = bf.generateSwaths(math.pi, mower.getCovWidth(), no_hl.getGeometry(0))
    snake_sorter = f2c.RP_Snake()
    swaths = snake_sorter.genSortedSwaths(swaths)

    path_planner = f2c.PP_PathPlanning()
    # dubins_cc = f2c.PP_DubinsCurvesCC()
    # path_dubins_cc = path_planner.planPath(mower, swaths, dubins_cc)
    reeds_shepp_hc = f2c.PP_ReedsSheppCurvesHC()
    path_reeds_shepp_hc = path_planner.planPath(mower, swaths, reeds_shepp_hc)
    drawCell([cell, swaths, no_hl, path_reeds_shepp_hc])
    # print(cell[0].area())
    # print(path_dubins_cc)


#
# def genPoints(num, origin: f2c.Point):
#     points = []
#     for i in range(num):
#         randX = random.randint(origin.getX() + 1, origin.getX() + 200)
#         randY = random.randint(origin.getY() + 1, origin.getY() + 200)
#         points.append(f2c.Point(randX, randY))
#     return points
#
# def sortPoints(points, origin):
#     hull = [origin]
#
#     points.sort(key=lambda x: x.Y)
#
#     origin = hull[0]
#     for i in range(len(points)):
#         points[i].angle = calcAngle(origin, points[i])
#
#     points.sort(key=lambda x: x.angle)
#     printPoints(points)
#     hull += points
#     return hull
#


def FieldConfig(angles, holes):
    # genPoints(5, f2c.Point(0, 0))
    cells = f2c.Cells(
        f2c.Cell(
            f2c.LinearRing(
                f2c.VectorPoint(
                    [
                        f2c.Point(0, 0),
                        f2c.Point(60, 0),
                        f2c.Point(60, 60),
                        f2c.Point(0, 60),
                        f2c.Point(0, 0),
                        # random.randint(0, 150),
                    ]
                )
            )
        )
    )
    cells.addRing(
        0,
        f2c.LinearRing(
            f2c.VectorPoint(
                [
                    f2c.Point(12, 12),
                    f2c.Point(12, 17),
                    f2c.Point(10, 20),
                    f2c.Point(19, 14),
                    f2c.Point(12, 12),
                ]
            )
        ),
    )

    # rand = f2c.Random()
    # field = rand.generateRandField(100, 5)
    # hole1 = field.getField()
    # print("hole1 for i ")
    # print("HOLE!")
    # print(hole1[0])
    #
    # # ring = f2c.LinearRing()
    # # for j in hole1:
    # #     ring.addPoint(j[0], j[1])
    #
    # cells.addGeometry(hole1[0])
    #
    # cells.addRing(ring)
    cells.addRing(
        0,
        f2c.LinearRing(
            f2c.VectorPoint(
                [
                    f2c.Point(36, 36),
                    f2c.Point(36, 48),
                    f2c.Point(48, 48),
                    f2c.Point(48, 36),
                    f2c.Point(36, 36),
                ]
            )
        ),
    )
    return cells


def randomFieldConfig(angles, holes):
    rand = f2c.Random()
    cells = f2c.Cells()

    field = rand.generateRandField(5e3, angles)
    fieldField = field.getField()
    cells.addGeometry(fieldField[0])

    for j in range(1):
        # rand = f2c.Random()
        hole = rand.generateRandCell(6e3, 3)
        for i in hole[0]:
            i.setX(i.getX() + 40)
        # holeField = hole.getField()
        cells.addGeometry(hole)

    return cells


def randomMultiHoles():
    mower = mowerConfig(1.0, 1.0)

    # rand = f2c.Random()
    # cells = f2c.Cells()
    #
    # field = rand.generateRandField(1e4, 5)
    # field1 = field.getField()
    # cells.addGeometry(field1[0])

    # for j in range(3):
    #     field = rand.generateRandField(1e4, 5)
    #     field = field.getField()
    #     cells.addGeometry(field[0])
    cells = FieldConfig(5, 4)

    const_hl = f2c.HG_Const_gen()
    mid_hl = const_hl.generateHeadlands(cells, 1.5 * mower.getWidth())
    no_hl = const_hl.generateHeadlands(cells, 3.0 * mower.getWidth())

    bf = f2c.SG_BruteForce()
    swaths = bf.generateSwaths(math.pi / 2.0, mower.getCovWidth(), no_hl)

    route_planner = f2c.RP_RoutePlannerBase()
    route = route_planner.genRoute(mid_hl, swaths)

    drawCell([cells, swaths, no_hl, route])
    drawCell(cells)
    print(cells[0].area())
    print(route)


# main()
# randomMultiHoles()
main()
