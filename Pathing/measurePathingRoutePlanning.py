import time
import math
import os
import matplotlib.pyplot as plt
import statistics
import Mapping as m
import fields2cover as f2c
from utils import mowerConfig, load_csv_points, genField, drawCell


def main():
    # x = [5, 10, 20, 30, 40, 50, 60, 80, 100, 125, 150, 175, 200]
    # x = [50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]
    # x = [1, 5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200]

    # x = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x = [0, 1, 2, 3]
    times = []

    hull = []
    for i in x:
        thisTimes = []
        for j in range(5):
            startTime = time.perf_counter()

            if i == 0:
                rand = f2c.Random(42)
                mower = mowerConfig(0.22, 0.15)
                const_hl = f2c.HG_Const_gen()
                cell = rand.generateRandField(1e3, 6)
                cells = cell.getField()
                const_hl = f2c.HG_Const_gen()
                mid_hl = const_hl.generateHeadlands(cells, 1.5 * mower.getWidth())
                no_hl = const_hl.generateHeadlands(cells, 3.0 * mower.getWidth())

                bf = f2c.SG_BruteForce()
                swaths = bf.generateSwaths(math.pi / 2.0, mower.getCovWidth(), no_hl)
                route_planner = f2c.RP_RoutePlannerBase()
                swaths = route_planner.genRoute(mid_hl, swaths)

            elif i == 1:
                rand = f2c.Random(42)
                robot = mowerConfig(0.22, 0.15)
                const_hl = f2c.HG_Const_gen()
                field = rand.generateRandField(1e3, 6)
                cells = field.getField()
                no_hl = const_hl.generateHeadlands(cells, 3.0 * robot.getWidth())
                bf = f2c.SG_BruteForce()
                swaths = bf.generateSwaths(
                    math.pi, robot.getCovWidth(), no_hl.getGeometry(0)
                )

                boustrophedon_sorter = f2c.RP_Boustrophedon()
                swaths = boustrophedon_sorter.genSortedSwaths(swaths)

            elif i == 2:
                rand = f2c.Random(42)
                robot = mowerConfig(0.22, 0.15)
                const_hl = f2c.HG_Const_gen()
                field = rand.generateRandField(1e3, 6)
                cells = field.getField()
                no_hl = const_hl.generateHeadlands(cells, 3.0 * robot.getWidth())
                bf = f2c.SG_BruteForce()
                swaths = bf.generateSwaths(
                    math.pi, robot.getCovWidth(), no_hl.getGeometry(0)
                )
                snake_sorter = f2c.RP_Snake()
                swaths = snake_sorter.genSortedSwaths(swaths)

            elif i == 3:
                rand = f2c.Random(42)
                robot = mowerConfig(0.22, 0.15)
                const_hl = f2c.HG_Const_gen()
                field = rand.generateRandField(1e3, 6)
                cells = field.getField()
                no_hl = const_hl.generateHeadlands(cells, 3.0 * robot.getWidth())
                bf = f2c.SG_BruteForce()
                swaths = bf.generateSwaths(
                    math.pi, robot.getCovWidth(), no_hl.getGeometry(0)
                )
                spiral_sorter = f2c.RP_Spiral(6)
                swaths = spiral_sorter.genSortedSwaths(swaths)

            # Plan path
            path_planner = f2c.PP_PathPlanning()
            dubins_cc = f2c.PP_DubinsCurves()
            path_dubins_cc = path_planner.planPath(mower, swaths, dubins_cc)

            endTime = time.perf_counter()
            thisTimes.append((endTime - startTime) * 1000)

        times.append(statistics.mean(thisTimes))

    plt.figure(figsize=(10, 10))
    plt.rcParams.update({"font.size": 16})
    plt.rcParams["savefig.directory"] = os.path.expanduser(
        "~/Programming/RobotMower/finalReport/images/"
    )
    # plt.scatter(x, times, color="blue", marker="o", s=100, alpha=0.7)

    pathPlanning = [
        "Shortest Route",
        "Boustrophedon Order",
        "Snake Order",
        "Spiral Order",
    ]

    bars = plt.bar(pathPlanning, times, color="blue", alpha=0.7, width=0.6)

    for bar in bars:
        height = bar.get_height()
        plt.gca().annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # plt.xlabel("Range on points")
    plt.xlabel("Path Planning Method Used")
    plt.ylabel("Execution Time (milliseconds)")
    # plt.title("Performance of map generation algorithm baseline with 3 holes")
    # plt.title("Map Generation Runtime vs Number of Holes")

    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
