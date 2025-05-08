import statistics
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
    mower = mowerConfig(0.22, 0.15)
    rand = f2c.Random(42)
    field_size = max(pow(10, i), 2.0)
    # print(field_size)
    field = rand.generateRandField(field_size, 6)
    cell = field.getField()

    const_hl = f2c.HG_Const_gen()
    no_hl = const_hl.generateHeadlands(cell, 3 * mower.getWidth())
    mid_hl = const_hl.generateHeadlands(cell, 1.5 * mower.getWidth())

    bf = f2c.SG_BruteForce()
    swaths = bf.generateSwaths(math.pi, mower.getCovWidth(), no_hl.getGeometry(0))

    boustrophedon_sorter = f2c.RP_Boustrophedon()
    swaths = boustrophedon_sorter.genSortedSwaths(swaths)

    path_planner = f2c.PP_PathPlanning()
    dubins_cc = f2c.PP_DubinsCurves()
    path_dubins_cc = path_planner.planPath(mower, swaths, dubins_cc)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Force cleanup
    del mower, rand, field, cell, const_hl, no_hl, mid_hl, bf, swaths
    del boustrophedon_sorter, path_planner, dubins_cc, path_dubins_cc
    gc.collect()

    return peak / 1024  # KB


def main():
    # x = list(range(1, 12))
    x = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    memory_usages = []

    for i in x:
        thisMemory = []
        # Use a process pool to ensure isolation between runs
        with mp.Pool(processes=1) as pool:
            for j in range(5):
                memory = pool.apply(run_test, (i,))
                thisMemory.append(memory)

        memory_usages.append(statistics.mean(thisMemory))

    # Print results
    # memory_usages.pop(0)
    # x.pop(10)
    for xi, avg_memory in zip(x, memory_usages):
        print(f"x = {xi}: average peak memory = {avg_memory:.2f} KB")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(x, memory_usages, color="blue", marker="o", s=100, alpha=0.7)
    plt.xlabel("Area of Field (10^xm^2)")
    plt.ylabel("Peak Memory Usage (KB)")
    plt.title("Path Planning Peak Memory Usage Compared to Size of Field")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("memory_usage.png")  # Save figure before showing
    plt.show()


if __name__ == "__main__":
    main()

