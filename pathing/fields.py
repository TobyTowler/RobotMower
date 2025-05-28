import math
import fields2cover as f2c
from utils import mowerConfig


# def main():
def makeHoles(i):
    # robot_c = f2c.Robot(1)
    # print("####### Tutorial 5.1 Route planning for all swaths ######")
    # i = 1
    # robot_c = f2c.Robot(0.22, 0.15)
    # cells_c = f2c.Cells(
    #     f2c.Cell(
    #         f2c.LinearRing(
    #             f2c.VectorPoint(
    #                 [
    #                     f2c.Point(0, 0),
    #                     f2c.Point(75, 0),
    #                     f2c.Point(75, 75),
    #                     f2c.Point(0, 75),
    #                     f2c.Point(0, 0),
    #                 ]
    #             )
    #         )
    #     )
    # )
    # if i > 0:
    #     cells_c.addRing(
    #         0,
    #         f2c.LinearRing(
    #             f2c.VectorPoint(
    #                 [
    #                     f2c.Point(12, 12),
    #                     f2c.Point(12, 18),
    #                     f2c.Point(18, 18),
    #                     f2c.Point(18, 12),
    #                     f2c.Point(12, 12),
    #                 ]
    #             )
    #         ),
    #     )
    # if i > 1:
    #     cells_c.addRing(
    #         0,
    #         f2c.LinearRing(
    #             f2c.VectorPoint(
    #                 [
    #                     f2c.Point(36, 36),
    #                     f2c.Point(36, 48),
    #                     f2c.Point(48, 48),
    #                     f2c.Point(48, 36),
    #                     f2c.Point(36, 36),
    #                 ]
    #             )
    #         ),
    #     )

    cells_c = f2c.Cells(
        f2c.Cell(
            f2c.LinearRing(
                f2c.VectorPoint(
                    [
                        f2c.Point(0, 0),
                        f2c.Point(80, 0),
                        f2c.Point(80, 80),
                        f2c.Point(0, 80),
                        f2c.Point(0, 0),
                    ]
                )
            )
        )
    )

    # Hole 1 (i > 0)
    if i > 0:
        cells_c.addRing(
            0,
            f2c.LinearRing(
                f2c.VectorPoint(
                    [
                        f2c.Point(10, 10),
                        f2c.Point(10, 20),
                        f2c.Point(20, 20),
                        f2c.Point(20, 10),
                        f2c.Point(10, 10),
                    ]
                )
            ),
        )

    # Hole 2 (i > 1)
    if i > 1:
        cells_c.addRing(
            0,
            f2c.LinearRing(
                f2c.VectorPoint(
                    [
                        f2c.Point(35, 10),
                        f2c.Point(35, 20),
                        f2c.Point(45, 20),
                        f2c.Point(45, 10),
                        f2c.Point(35, 10),
                    ]
                )
            ),
        )

    # Hole 3 (i > 2)
    if i > 2:
        cells_c.addRing(
            0,
            f2c.LinearRing(
                f2c.VectorPoint(
                    [
                        f2c.Point(60, 10),
                        f2c.Point(60, 20),
                        f2c.Point(70, 20),
                        f2c.Point(70, 10),
                        f2c.Point(60, 10),
                    ]
                )
            ),
        )

    # Hole 4 (i > 3)
    if i > 3:
        cells_c.addRing(
            0,
            f2c.LinearRing(
                f2c.VectorPoint(
                    [
                        f2c.Point(10, 35),
                        f2c.Point(10, 45),
                        f2c.Point(20, 45),
                        f2c.Point(20, 35),
                        f2c.Point(10, 35),
                    ]
                )
            ),
        )

    # Hole 5 (i > 4)
    if i > 4:
        cells_c.addRing(
            0,
            f2c.LinearRing(
                f2c.VectorPoint(
                    [
                        f2c.Point(35, 35),
                        f2c.Point(35, 45),
                        f2c.Point(45, 45),
                        f2c.Point(45, 35),
                        f2c.Point(35, 35),
                    ]
                )
            ),
        )

    # Hole 6 (i > 5)
    if i > 5:
        cells_c.addRing(
            0,
            f2c.LinearRing(
                f2c.VectorPoint(
                    [
                        f2c.Point(60, 35),
                        f2c.Point(60, 45),
                        f2c.Point(70, 45),
                        f2c.Point(70, 35),
                        f2c.Point(60, 35),
                    ]
                )
            ),
        )

    # Hole 7 (i > 6)
    if i > 6:
        cells_c.addRing(
            0,
            f2c.LinearRing(
                f2c.VectorPoint(
                    [
                        f2c.Point(10, 60),
                        f2c.Point(10, 70),
                        f2c.Point(20, 70),
                        f2c.Point(20, 60),
                        f2c.Point(10, 60),
                    ]
                )
            ),
        )

    # Hole 8 (i > 7)
    if i > 7:
        cells_c.addRing(
            0,
            f2c.LinearRing(
                f2c.VectorPoint(
                    [
                        f2c.Point(35, 60),
                        f2c.Point(35, 70),
                        f2c.Point(45, 70),
                        f2c.Point(45, 60),
                        f2c.Point(35, 60),
                    ]
                )
            ),
        )

    # Hole 9 (i > 8)
    if i > 8:
        cells_c.addRing(
            0,
            f2c.LinearRing(
                f2c.VectorPoint(
                    [
                        f2c.Point(60, 60),
                        f2c.Point(60, 70),
                        f2c.Point(70, 70),
                        f2c.Point(70, 60),
                        f2c.Point(60, 60),
                    ]
                )
            ),
        )

    # Hole 10 (i > 9) - centered in the field
    # if i > 9:
    #     cells_c.addRing(
    #         0,
    #         f2c.LinearRing(
    #             f2c.VectorPoint(
    #                 [
    #                     f2c.Point(32, 32),
    #                     f2c.Point(32, 48),
    #                     f2c.Point(48, 48),
    #                     f2c.Point(48, 32),
    #                     f2c.Point(32, 32),
    #                 ]
    #             )
    #         ),
    #     )

    return cells_c

    const_hl = f2c.HG_Const_gen()
    mid_hl_c = const_hl.generateHeadlands(cells_c, 1.5 * robot_c.getWidth())
    no_hl_c = const_hl.generateHeadlands(cells_c, 3.0 * robot_c.getWidth())
    bf = f2c.SG_BruteForce()
    swaths_c = bf.generateSwaths(math.pi / 2.0, robot_c.getCovWidth(), no_hl_c)
    route_planner = f2c.RP_RoutePlannerBase()
    route = route_planner.genRoute(mid_hl_c, swaths_c)
    # return cell

    # f2c.Visualizer.figure()
    # f2c.Visualizer.plot(cells_c)
    # f2c.Visualizer.plot(no_hl_c)
    # f2c.Visualizer.xlim(-5, 210)
    # f2c.Visualizer.ylim(-5, 210)
    # # f2c.Visualizer.show()
    # f2c.Visualizer.figure()
    # f2c.Visualizer.plot(cells_c)
    # f2c.Visualizer.plot(no_hl_c)
    # f2c.Visualizer.plot(route)
    # # f2c.Visualizer.xlim(-5, 65)
    # # f2c.Visualizer.ylim(-5, 65)
    # f2c.Visualizer.show()
    # # print("####### Tutorial 5.2 Known Patterns ######")


def main():
    mower = mowerConfig(0.22, 0.15)

    cell = makeHoles(3)

    const_hl = f2c.HG_Const_gen()
    mid_hl_c = const_hl.generateHeadlands(cell, 1.5 * mower.getWidth())
    no_hl_c = const_hl.generateHeadlands(cell, 0 * mower.getWidth())
    bf = f2c.SG_BruteForce()
    swaths_c = bf.generateSwaths(math.pi / 2.0, mower.getCovWidth(), no_hl_c)
    route_planner = f2c.RP_RoutePlannerBase()
    route = route_planner.genRoute(mid_hl_c, swaths_c)
    # makeHoles(10)


if __name__ == "__main__":
    main()
