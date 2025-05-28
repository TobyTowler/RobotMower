import time
import math
import os
import matplotlib.pyplot as plt
import statistics
import Mapping as m
import fields2cover as f2c
from utils import mowerConfig, load_csv_points, genField, drawCell
import multiprocessing as mp
import tracemalloc
from fields import makeHoles
import gc


def main():
    x = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    times = []

    for i in x:
        thisTimes = []
        for j in range(5):
            print(i, " ", j)
            tracemalloc.start()

            mower = mowerConfig(0.44, 0.30)

            cell = makeHoles(i)

            const_hl = f2c.HG_Const_gen()
            mid_hl_c = const_hl.generateHeadlands(cell, 1.5 * mower.getWidth())
            no_hl_c = const_hl.generateHeadlands(cell, 0 * mower.getWidth())
            bf = f2c.SG_BruteForce()
            swaths_c = bf.generateSwaths(math.pi / 2.0, mower.getCovWidth(), no_hl_c)
            route_planner = f2c.RP_RoutePlannerBase()
            route = route_planner.genRoute(mid_hl_c, swaths_c)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Force cleanup
            # del mower, rand, field, cell, const_hl, no_hl, mid_hl, bf, swaths
            # del boustrophedon_sorter, path_planner, dubins_cc, path_dubins_cc
            gc.collect()

            thisTimes.append(peak / 1024)  # KB

        times.append(statistics.mean(thisTimes))

    times.pop(0)
    x.pop(0)

    plt.figure(figsize=(10, 10))
    plt.rcParams.update({"font.size": 16})
    plt.rcParams["savefig.directory"] = os.path.expanduser(
        "~/Programming/RobotMower/finalReport/images/"
    )
    plt.scatter(x, times, color="blue", marker="o", s=100, alpha=0.7)
    plt.xlabel("Number of Holes")
    plt.ylabel("Peak Memory Usage (KB)")

    # bars = plt.bar(pathPlanning, times, color="blue", alpha=0.7, width=0.6)

    # for bar in bars:
    #     height = bar.get_height()
    #     plt.gca().annotate(
    #         f"{height:.2f}",
    #         xy=(bar.get_x() + bar.get_width() / 2, height),
    #         xytext=(0, 3),  # 3 points vertical offset
    #         textcoords="offset points",
    #         ha="center",
    #         va="bottom",
    #         fontsize=10,
    #         fontweight="bold",
    #     )

    # plt.title("Path Planning Peak Memory Usage Compared to Size of Field")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    # plt.savefig("memory_usage.png")  # Save figure before showing
    plt.show()


if __name__ == "__main__":
    main()
