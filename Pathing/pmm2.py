import statistics
import math
import fields2cover as f2c
import tracemalloc
import Mapping as m
import matplotlib.pyplot as plt
from utils import mowerConfig, load_csv_points, genField, drawCell
import sys
import gc


def reset_caches():
    # Force garbage collection
    gc.collect()

    # Clear module caches that you're specifically using
    modules_to_reload = ["fields2cover", "Mapping", "utils"]
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            del sys.modules[module_name]


def test():
    tracemalloc.start()
    mower = mowerConfig(0.22, 0.15)

    # Use a fixed seed for stability
    rand = f2c.Random(42)

    # Use a minimum field size to avoid "Geometry does not contain point 0" error
    # The library seems to struggle with very small fields
    field_size = max(pow(10, 3), 2.0)  # Ensure minimum size of 2.0

    # Generate field with more vertices for small fields
    field = rand.generateRandField(field_size, 6)
    cell = field.getField()

    # Generate headlands with smaller width for small fields
    const_hl = f2c.HG_Const_gen()
    # Scale headland width based on field size to avoid issues
    # width_factor = min(
    #     3.0, max(0.5, 0.5 + i)
    # )  # Scale from 0.5 to 3.0 based on field size
    # width_val = width_factor * mower.getWidth()
    no_hl = const_hl.generateHeadlands(cell, 3 * mower.getWidth())
    mid_hl = const_hl.generateHeadlands(cell, 1.5 * mower.getWidth())
    # Generate swaths with appropriate parameters
    bf = f2c.SG_BruteForce()
    swaths = bf.generateSwaths(math.pi, mower.getCovWidth(), no_hl.getGeometry(0))

    boustrophedon_sorter = f2c.RP_Boustrophedon()
    swaths = boustrophedon_sorter.genSortedSwaths(swaths)

    path_planner = f2c.PP_PathPlanning()
    dubins_cc = f2c.PP_DubinsCurves()
    path_dubins_cc = path_planner.planPath(mower, swaths, dubins_cc)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()
    return peak


def main():
    # x = [1, 5, 10, 20, 30, 40, 50, 75, 100, 120, 150, 200]
    # x = [5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200]
    # x = [50, 200, 400, 600, 800, 1000, 1200, 1400]
    x = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
    ]
    memory_usages = []

    for i in x:
        thisMemory = []
        for j in range(5):
            thisMemory.append(test() / 1024)  # KB

        memory_usages.append(statistics.mean(thisMemory))

    # Print results
    for xi, avg_memory in zip(x, memory_usages):
        print(f"x = {xi}: average peak memory = {avg_memory:.2f} KB")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(x, memory_usages, color="blue", marker="o", s=100, alpha=0.7)

    plt.xlabel("Number of Runs")
    plt.ylabel("Peak Memory Usage (KB)")
    # plt.title("Peak Memory Usage vs Number of Holes in Map Generation")
    plt.title("Path Planning Memory Usage Baseline")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
