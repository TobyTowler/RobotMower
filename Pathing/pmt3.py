import time
import os
import matplotlib.pyplot as plt
import statistics
import fields2cover as f2c
import math
import gc
from utils import mowerConfig, load_csv_points, genField, drawCell


def process_single_run(i, j):
    """Perform a single timing run safely"""
    try:
        startTime = time.perf_counter()

        # Create new objects for each run to avoid memory conflicts
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

        # Sort swaths

        # if i == 0:
        #     const_hl = f2c.HG_Const_gen()
        #     mid_hl = const_hl.generateHeadlands(cell, 1.5 * mower.getWidth())
        #     no_hl = const_hl.generateHeadlands(cell, 3.0 * mower.getWidth())
        #     bf = f2c.SG_BruteForce()
        #     swaths = bf.generateSwaths(math.pi / 2.0, mower.getCovWidth(), no_hl)
        #     route_planner = f2c.RP_RoutePlannerBase()
        #     swaths = route_planner.genRoute(mid_hl, swaths)
        #
        # elif i == 1:
        #     boustrophedon_sorter = f2c.RP_Boustrophedon()
        #     swaths = boustrophedon_sorter.genSortedSwaths(swaths)
        #
        # elif i == 2:
        #     snake_sorter = f2c.RP_Snake()
        #     swaths = snake_sorter.genSortedSwaths(swaths)
        #
        # elif i == 3:
        #     spiral_sorter = f2c.RP_Spiral(6)
        #     swaths = spiral_sorter.genSortedSwaths(swaths)

        # Plan path
        path_planner = f2c.PP_PathPlanning()
        dubins_cc = f2c.PP_DubinsCurves()
        path_dubins_cc = path_planner.planPath(mower, swaths, dubins_cc)
        endTime = time.perf_counter()

        # Explicitly clean up to avoid memory issues
        del field, cell, no_hl, swaths, path_dubins_cc
        gc.collect()

        return (endTime - startTime) * 1000
    except Exception as e:
        print(f"Error processing run {i},{j}: {e}")
        # Force garbage collection
        gc.collect()
        return None


pathPlanning = ["Shortest Route", "Boustrophedon Order", "Snake Order", "Spiral Order"]


def main():
    # Use indices that match pathPlanning list positions
    x = [0, 1, 2, 3]
    times = []

    # Process each path planning method
    for i in x:
        print(f"Processing path planning method: {pathPlanning[i]}")
        thisTimes = []
        # Run 5 iterations for each method
        for j in range(5):
            print(f"  Run {j + 1}/5...", end="", flush=True)
            result = process_single_run(i, j)
            if result is not None:
                thisTimes.append(result)
                print(f" completed in {result:.2f} ms")
            else:
                print(" failed")
            # Add a small delay between runs to let memory clean up
            time.sleep(0.2)
        if thisTimes:
            avg_time = statistics.mean(thisTimes)
            times.append(avg_time)
            print(f"Average time for {pathPlanning[i]}: {avg_time:.2f} ms")
        else:
            times.append(float("nan"))
            print(f"All runs failed for {pathPlanning[i]}")

    # Plot results if we have any valid data
    if any(not math.isnan(t) for t in times):
        plt.figure(figsize=(10, 6))
        # Filter out NaN values for plotting
        valid_indices = []
        valid_times = []
        for idx, t in enumerate(times):
            if not math.isnan(t):
                valid_indices.append(idx)
                valid_times.append(t)

        # Plot with the string labels from pathPlanning
        valid_labels = [pathPlanning[i] for i in valid_indices]

        plt.bar(valid_labels, valid_times, color="blue", alpha=0.7, width=0.6)

        plt.figure(figsize=(10, 10))
        plt.rcParams.update({"font.size": 16})
        plt.rcParams["savefig.directory"] = os.path.expanduser(
            "~/Programming/RobotMower/finalReport/images/"
        )

        # Customize the plot
        plt.xlabel("Route Planning Method")
        plt.ylabel("Execution Time (milliseconds)")
        # plt.title("Path Planning Runtime vs Route Planning Method")
        plt.grid(True, linestyle="--", alpha=0.7, axis="y")
        plt.xticks(rotation=15)  # Rotate labels slightly for better readability

        # Add data values on top of each bar
        for i, v in enumerate(valid_times):
            plt.text(i, v + 0.5, f"{v:.2f}", ha="center")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
