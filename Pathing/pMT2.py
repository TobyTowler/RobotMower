import time
import matplotlib.pyplot as plt
import statistics
import fields2cover as f2c
import math
import concurrent.futures
import multiprocessing
import numpy as np
from utils import mowerConfig, load_csv_points, genField, drawCell


def process_task(i_j):
    """Process a single computation with preloaded parameters"""
    i, j = i_j
    startTime = time.perf_counter()

    # Initialize once
    mower = mowerConfig(0.22, 1.5)

    # Use different seed per run to avoid contention
    rand = f2c.Random(42 + j)

    # Use efficient power calculation
    field_size = int(pow(10, i))
    field = rand.generateRandField(field_size, 6)
    cell = field.getField()

    # Precompute constants
    width_multiplier = 3.0 * mower.getWidth()
    const_hl = f2c.HG_Const_gen()
    no_hl = const_hl.generateHeadlands(cell, width_multiplier)

    # Use optimized swath generation
    bf = f2c.SG_BruteForce()
    cov_width = mower.getCovWidth()
    geom = no_hl.getGeometry(0)
    swaths = bf.generateSwaths(math.pi, cov_width, geom)

    # Apply snake sorting
    snake_sorter = f2c.RP_Snake()
    swaths = snake_sorter.genSortedSwaths(swaths)

    # Plan path efficiently
    path_planner = f2c.PP_PathPlanning()
    dubins = f2c.PP_DubinsCurvesCC()
    path_dubins = path_planner.planPath(mower, swaths, dubins)

    endTime = time.perf_counter()
    return (endTime - startTime) * 1000


def main():
    # Define range of field sizes to test
    x = [1, 2, 3]
    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    times = []

    # Determine optimal number of workers
    # Use 75% of available cores or max 4, whichever is smaller
    num_cores = min(max(1, int(multiprocessing.cpu_count() * 0.75)), 4)

    # Process each field size
    for i in x:
        # Generate all parameter combinations
        tasks = [(i, j) for j in range(5)]
        results = []

        # Use threading since Fields2Cover has native C++ parallelism
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(process_task, task): task for task in tasks
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Task {task} generated an exception: {e}")

        # Calculate mean execution time
        if results:
            times.append(statistics.mean(results))
        else:
            times.append(float("nan"))

    # Create plot with optimized settings
    plt.figure(figsize=(8, 6))
    plt.scatter(x, times, color="blue", marker="o", s=100, alpha=0.7)

    plt.xlabel("Size of field 10^x")
    plt.ylabel("Execution Time (milliseconds)")
    plt.title("Path Planning Runtime vs Size of Field")

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save plot to file and show only if needed
    plt.savefig("performance_results.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
