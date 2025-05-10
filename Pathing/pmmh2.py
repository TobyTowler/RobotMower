import time
import os
import matplotlib.pyplot as plt
import statistics
import fields2cover as f2c
import math
import gc
from fields import makeHoles
from utils import mowerConfig, load_csv_points, genField, drawCell
import tracemalloc
import psutil
import sys


def measure_memory_usage(i):
    """Measure memory usage using psutil instead of tracemalloc"""
    process = psutil.Process()

    # Get initial memory
    mem_before = process.memory_info().rss

    try:
        mower = mowerConfig(0.22, 0.15)
        cell = makeHoles(i)

        const_hl = f2c.HG_Const_gen()
        mid_hl_c = const_hl.generateHeadlands(cell, 1.5 * mower.getWidth())
        no_hl_c = const_hl.generateHeadlands(cell, 0 * mower.getWidth())
        bf = f2c.SG_BruteForce()
        swaths_c = bf.generateSwaths(math.pi / 2.0, mower.getCovWidth(), no_hl_c)
        route_planner = f2c.RP_RoutePlannerBase()
        route = route_planner.genRoute(mid_hl_c, swaths_c)

        # Get peak memory during operation
        mem_after = process.memory_info().rss

        # Clean up
        del cell, no_hl_c, swaths_c, route
        gc.collect()

        return (mem_after - mem_before) / 1024  # Convert to KB

    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        gc.collect()


def process_single_run_safe(i, j):
    """Safer version using psutil instead of tracemalloc"""
    try:
        # Use psutil for memory monitoring
        process = psutil.Process()
        mem_before = process.memory_info().rss

        mower = mowerConfig(0.22, 0.15)
        cell = makeHoles(i)

        const_hl = f2c.HG_Const_gen()
        mid_hl_c = const_hl.generateHeadlands(cell, 1.5 * mower.getWidth())
        no_hl_c = const_hl.generateHeadlands(cell, 0 * mower.getWidth())

        # Force garbage collection between operations
        gc.collect()

        bf = f2c.SG_BruteForce()
        swaths_c = bf.generateSwaths(math.pi / 2.0, mower.getCovWidth(), no_hl_c)

        gc.collect()

        route_planner = f2c.RP_RoutePlannerBase()
        route = route_planner.genRoute(mid_hl_c, swaths_c)

        # Get peak memory
        mem_after = process.memory_info().rss
        peak_memory = mem_after - mem_before

        # Clean up immediately
        del cell, no_hl_c, swaths_c, route
        del mower, const_hl, mid_hl_c, bf, route_planner
        gc.collect()

        return peak_memory / 1024  # Convert to KB

    except Exception as e:
        print(f"Error processing run {i},{j}: {e}")
        gc.collect()
        return None


def main():
    # Reduce the test size to prevent freezing
    x = [0, 1, 2, 3, 4, 5]  # Test with fewer holes
    times = []

    # Set matplotlib backend to avoid display issues
    plt.switch_backend("Agg")

    # Process each number of holes
    for i in x:
        print(f"Processing {i} holes...")
        thisTimes = []

        # Run fewer iterations to prevent accumulation
        for j in range(3):  # Reduced from 5 to 3
            print(f"  Run {j + 1}/3...", end="", flush=True)

            # Monitor system memory
            if psutil.virtual_memory().percent > 85:
                print(
                    f" Skipping - System memory at {psutil.virtual_memory().percent}%"
                )
                continue

            result = process_single_run_safe(i, j)
            if result is not None:
                thisTimes.append(result)
                print(f" completed with {result:.2f} KB")
            else:
                print(" failed")

            # Longer delay between runs
            time.sleep(1)

            # Force garbage collection
            gc.collect()

        if thisTimes:
            avg_memory = statistics.mean(thisTimes)
            times.append(avg_memory)
            print(f"Average memory for {i} holes: {avg_memory:.2f} KB")
        else:
            times.append(float("nan"))
            print(f"All runs failed for {i} holes")

        # Longer delay between different hole counts
        time.sleep(2)

    # Plot results if we have any valid data
    if any(not math.isnan(t) for t in times):
        try:
            plt.figure(figsize=(10, 10))
            plt.rcParams.update({"font.size": 16})

            plt.scatter(x, times, color="blue", marker="o", s=100, alpha=0.7)
            plt.xlabel("Number of Holes")
            plt.ylabel("Peak Memory Usage (KB)")
            plt.grid(True, linestyle="--", alpha=0.7, axis="y")
            plt.xticks(x)  # Use actual x values for ticks

            plt.tight_layout()

            # Save instead of showing to prevent display issues
            save_path = os.path.expanduser(
                "~/Programming/RobotMower/finalReport/images/pathingHolesMem_safe.png"
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        except Exception as e:
            print(f"Error creating plot: {e}")
        finally:
            plt.close("all")


if __name__ == "__main__":
    # Add memory limit check
    if (
        psutil.virtual_memory().available < 1024 * 1024 * 1024
    ):  # Less than 1GB available
        print("Warning: Less than 1GB of memory available. Results may be unreliable.")

    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Final cleanup
        gc.collect()
