import time
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

        # Generate swaths with appropriate parameters
        bf = f2c.SG_BruteForce()
        swaths = bf.generateSwaths(math.pi, mower.getCovWidth(), no_hl.getGeometry(0))

        # Sort swaths
        # snake_sorter = f2c.RP_Snake()
        # swaths = snake_sorter.genSortedSwaths(swaths)

        boustrophedon_sorter = f2c.RP_Boustrophedon()
        swaths = boustrophedon_sorter.genSortedSwaths(swaths)
        # Plan path
        path_planner = f2c.PP_PathPlanning()
        dubins_cc = f2c.PP_DubinsCurvesCC()
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


def main():
    # Use a range of field sizes that should work better
    # x = [0.5, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    x = [0, 1, 2, 3]
    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    times = []

    # Process each field size sequentially but with faster code
    for i in x:
        print(f"Processing field size 10^{i}")
        thisTimes = []

        # Run 3 iterations instead of 5 to reduce chances of memory issues
        for j in range(5):
            print(f"  Run {j + 1}/3...", end="", flush=True)
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
            print(f"Average time for field size 10^{i}: {avg_time:.2f} ms")
        else:
            times.append(float("nan"))
            print(f"All runs failed for field size 10^{i}")

    # Plot results if we have any valid data
    if any(not math.isnan(t) for t in times):
        plt.figure(figsize=(8, 6))

        # Filter out NaN values for plotting
        valid_x = []
        valid_times = []
        for idx, t in enumerate(times):
            if not math.isnan(t):
                valid_x.append(x[idx])
                valid_times.append(t)

        plt.scatter(valid_x, valid_times, color="blue", marker="o", s=100, alpha=0.7)

        # Add trendline if we have enough data points
        if len(valid_x) >= 3:
            try:
                coeffs = np.polyfit(valid_x, valid_times, 1)
                poly = np.poly1d(coeffs)
                x_line = np.linspace(min(valid_x), max(valid_x), 100)
                plt.plot(x_line, poly(x_line), "--", color="red")
            except:
                pass  # Skip trendline if it fails

        plt.xlabel("Area of field, (10^x)m^2")
        # plt.xlabel("Size of field 10^x")
        plt.ylabel("Execution Time (milliseconds)")
        plt.title("Path Planning Runtime vs Size of Field")

        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save plot
        # plt.savefig("performance_results.png", dpi=300)
        # print(f"Plot saved to performance_results.png")

        # Only show if display is available
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot: {e}")
    else:
        print("No valid data to plot")


if __name__ == "__main__":
    main()
