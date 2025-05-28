import statistics
from fields import makeHoles
import math
import fields2cover as f2c
import tracemalloc
import Mapping as m
import matplotlib.pyplot as plt
from utils import mowerConfig, load_csv_points, genField, drawCell
import gc
import multiprocessing as mp
import os
import random

# Set environment variables to disable optimizations
os.environ["PYTHONMALLOC"] = "malloc"
os.environ["PYTHONHASHSEED"] = str(random.randint(1, 1000000))


def run_test(i):
    """Run a single memory test in isolation"""
    tracemalloc.start()

    # Your test code here
    # mower = mowerConfig(0.22, 0.15)
    # rand = f2c.Random(42)
    # # field_size = max(pow(10, i), 2.0)
    # # print(field_size)
    # # field = rand.generateRandField(field_size, 6)
    # field = rand.generateRandField(10e3, 6)
    # cell = field.getField()
    #
    # const_hl = f2c.HG_Const_gen()
    # no_hl = const_hl.generateHeadlands(cell, 3 * mower.getWidth())
    #
    # bf = f2c.SG_BruteForce()
    # swaths = bf.generateSwaths(math.pi, mower.getCovWidth(), no_hl.getGeometry(0))
    #
    # boustrophedon_sorter = f2c.RP_Boustrophedon()
    # swaths = boustrophedon_sorter.genSortedSwaths(swaths)
    #
    # path_planner = f2c.PP_PathPlanning()
    # dubins_cc = f2c.PP_DubinsCurves()
    # path_dubins_cc = path_planner.planPath(mower, swaths, dubins_cc)
    #
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
    # del mower, rand, field, cell, const_hl, no_hl, bf, swaths
    # del boustrophedon_sorter, path_planner, dubins_cc, path_dubins_cc
    del cell, mower, const_hl, mid_hl_c, no_hl_c, bf, swaths_c, route, route_planner
    gc.collect()

    return peak / 1024  # KB


def run_testPlanning(i):
    """Run a single memory test in isolation"""
    tracemalloc.start()

    mower = mowerConfig(0.44, 0.30)

    # cell = makeHoles(i)
    rand = f2c.Random()
    field = rand.generateRandField(10e3, 6)
    cell = field.getField()

    const_hl = f2c.HG_Const_gen()
    no_hl = const_hl.generateHeadlands(cell, 3.0 * mower.getWidth())
    bf = f2c.SG_BruteForce()

    if i == 0:
        mid_hl = const_hl.generateHeadlands(cell, 1.5 * mower.getWidth())
        swaths = bf.generateSwaths(math.pi / 2.0, mower.getCovWidth(), no_hl)
        route_planner = f2c.RP_RoutePlannerBase()
        swaths = route_planner.genRoute(mid_hl, swaths)

    elif i == 1:
        swaths = bf.generateSwaths(math.pi, mower.getCovWidth(), no_hl.getGeometry(0))
        boustrophedon_sorter = f2c.RP_Boustrophedon()
        swaths = boustrophedon_sorter.genSortedSwaths(swaths)

    elif i == 2:
        swaths = bf.generateSwaths(math.pi, mower.getCovWidth(), no_hl.getGeometry(0))
        snake_sorter = f2c.RP_Snake()
        swaths = snake_sorter.genSortedSwaths(swaths)

    elif i == 3:
        swaths = bf.generateSwaths(math.pi, mower.getCovWidth(), no_hl.getGeometry(0))
        spiral_sorter = f2c.RP_Spiral(6)
        swaths = spiral_sorter.genSortedSwaths(swaths)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Force cleanup
    del cell, mower, const_hl, rand, bf, field, no_hl
    gc.collect()

    return peak / 1024  # KB


def main():
    # x = list(range(1, 12))
    # x = [1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    # x = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    x = [1, 0, 1, 2, 3]
    memory_usages = []
    for i in x:
        thisMemory = []
        for j in range(2):
            # Use a process pool to ensure isolation between runs
            with mp.Pool(processes=1) as pool:
                print(i, j)
                memory = pool.apply(run_testPlanning, (i,))
                thisMemory.append(memory)

        memory_usages.append(statistics.mean(thisMemory))

    # Print results
    memory_usages.pop(0)
    x.pop(0)
    for xi, avg_memory in zip(x, memory_usages):
        print(f"x = {xi}: average peak memory = {avg_memory:.2f} KB")

    # Plotting

    plt.figure(figsize=(20, 20))
    plt.rcParams.update({"font.size": 25})
    # plt.rcParams["savefig.directory"] = os.path.expanduser(
    #     "~/Programming/RobotMower/finalReport/images/"
    # )
    # plt.scatter(x, memory_usages, color="blue", marker="o", s=100, alpha=0.7)

    pathPlanning = [
        "Shortest Route",
        "Boustrophedon Order",
        "Snake Order",
        "Spiral Order",
    ]

    bars = plt.bar(pathPlanning, memory_usages, color="blue", alpha=0.7, width=0.6)
    plt.xlabel("Route Planning Method")
    # plt.xlabel("Area of Field (10^xm^2)")
    plt.ylabel("Peak Memory Usage (KB)")

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

    # plt.title("Path Planning Peak Memory Usage Compared to Size of Field")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    # plt.savefig("memory_usage.png")  # Save figure before showing
    plt.show()


if __name__ == "__main__":
    main()
