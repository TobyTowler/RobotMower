import time
import matplotlib.pyplot as plt
import statistics
import fields2cover as f2c
import math
import gc
from fields import makeHoles
from utils import mowerConfig, load_csv_points, genField, drawCell


def process_single_run(i, j):
    """Perform a single timing run safely"""
    try:
        startTime = time.perf_counter()

        mower = mowerConfig(0.5, 0.8)

        cell = makeHoles(i)

        const_hl = f2c.HG_Const_gen()
        mid_hl_c = const_hl.generateHeadlands(cell, 1.5 * mower.getWidth())
        no_hl_c = const_hl.generateHeadlands(cell, 3.0 * mower.getWidth())
        bf = f2c.SG_BruteForce()
        swaths_c = bf.generateSwaths(math.pi / 2.0, mower.getCovWidth(), no_hl_c)
        route_planner = f2c.RP_RoutePlannerBase()
        route = route_planner.genRoute(mid_hl_c, swaths_c)

        # const_hl = f2c.HG_Const_gen()

        # no_hl = const_hl.generateHeadlands(cell, 3 * mower.getWidth())
        # mid_hl = const_hl.generateHeadlands(cell, 1.5 * mower.getWidth())
        #
        # bf = f2c.SG_BruteForce()
        # swaths = bf.generateSwaths(math.pi, mower.getCovWidth(), no_hl.getGeometry(0))
        #
        # boustrophedon_sorter = f2c.RP_Boustrophedon()
        # swaths = boustrophedon_sorter.genSortedSwaths(swaths)

        # const_hl = f2c.HG_Const_gen()
        # mid_hl = const_hl.generateHeadlands(cell, 1.5 * mower.getWidth())
        # no_hl = const_hl.generateHeadlands(cell, 3.0 * mower.getWidth())
        # bf = f2c.SG_BruteForce()
        # swaths = bf.generateSwaths(math.pi / 2.0, mower.getCovWidth(), no_hl)

        # bf = f2c.SG_BruteForce()
        # swaths = bf.generateSwaths(math.pi, mower.getCovWidth(), no_hl.getGeometry(0))
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

        # path_planner = f2c.PP_PathPlanning()
        # dubins_cc = f2c.PP_DubinsCurves()
        # path_dubins_cc = path_planner.planPath(mower, swaths, dubins_cc)

        # path_planner = f2c.PP_PathPlanning()
        # dubins_cc = f2c.PP_DubinsCurves()
        # path_dubins_cc = path_planner.planPath(mower, swaths, dubins_cc)
        endTime = time.perf_counter()

        # drawCell([cell, swaths, no_hl, path_dubins_cc])

        # Explicitly clean up to avoid memory issues
        del cell, no_hl_c, swaths_c, route
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
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # x = [
    #     0,
    #     1,
    # ]
    times = []

    # Process each path planning method
    for i in x:
        print(f"Processing path planning method: {i}")
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
            print(f"Average time for {x[i]}: {avg_time:.2f} ms")
        else:
            times.append(float("nan"))
            print(f"All runs failed for {x[i]}")

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
        # valid_labels = [pathPlanning[i] for i in valid_indices]

        # plt.bar(x[i], valid_times, color="blue", alpha=0.7, width=0.6)

        plt.scatter(x, times, color="blue", marker="o", s=100, alpha=0.7)
        plt.xlabel("Number of Holes")
        plt.ylabel("Execution Time (milliseconds)")
        plt.title("Path Planning Runtime vs Number of Holes")
        plt.grid(True, linestyle="--", alpha=0.7, axis="y")
        plt.xticks(rotation=15)

        plt.rcParams["text.usetex"] = False  # Disable TeX
        plt.rcParams["text.color"] = "none"  # Hide text

        # Add data values on top of each bar
        for i, v in enumerate(valid_times):
            plt.text(i, v + 0.5, f"{v:.2f}", ha="center")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
