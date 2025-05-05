import time
import matplotlib.pyplot as plt
import statistics
import fields2cover as f2c
import math
from utils import mowerConfig, load_csv_points, genField, drawCell


def main():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    times = []

    for i in x:
        thisTimes = []
        for j in range(5):
            startTime = time.perf_counter()
            mower = mowerConfig(0.22, 0.15)

            rand = f2c.Random(42)
            field = rand.generateRandField(1e4, 6)
            # hole = rand.generateRandCell(121, 4)
            # # hole1 = hole.getField()
            cell = field.getField()

            const_hl = f2c.HG_Const_gen()
            no_hl = const_hl.generateHeadlands(cell, 3.0 * mower.getWidth())

            bf = f2c.SG_BruteForce()
            swaths = bf.generateSwaths(
                math.pi, mower.getCovWidth(), no_hl.getGeometry(0)
            )
            snake_sorter = f2c.RP_Snake()
            swaths = snake_sorter.genSortedSwaths(swaths)

            path_planner = f2c.PP_PathPlanning()
            dubins = f2c.PP_DubinsCurvesCC()
            path_dubins = path_planner.planPath(mower, swaths, dubins)
            # drawCell([cell, swaths, no_hl, path_dubins])

            endTime = time.perf_counter()
            thisTimes.append((endTime - startTime) * 1000)

        times.append(statistics.mean(thisTimes))

    plt.figure(figsize=(8, 6))
    plt.scatter(x, times, color="blue", marker="o", s=100, alpha=0.7)

    # plt.xlabel("Range on points")
    plt.xlabel("Number of Runs")
    plt.ylabel("Execution Time (milliseconds)")
    # plt.title("Performance of map generation algorithm baseline with 3 holes")
    plt.title("Path Planning Runtime Baseline")

    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
