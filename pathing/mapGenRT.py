import statistics
import os
import time
import Mapping as m
import matplotlib.pyplot as plt


def run_runtime_test(x_values):
    """Run runtime test for given x values"""
    runtime_usages = []
    for i in x_values:
        thisRuntime = []
        for j in range(5):  # Keep original 5 runs
            start_time = time.time()
            hull = []  # move inside to avoid growing across tests
            origin = m.Point(2, 2)
            field = m.genPoints(10, origin, 400)  # Keep original parameters
            hull.append(m.sortPoints(field, origin))
            end_time = time.time()
            thisRuntime.append(
                (end_time - start_time) * 1000
            )  # Convert to milliseconds
        runtime_usages.append(statistics.mean(thisRuntime))
    return runtime_usages


def main():
    # Different x arrays for each test
    x_arrays = [
        [
            1,
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
            # 11,
            # 12,
            # 13,
            # 14,
            # 15,
            # 16,
            # 17,
            # 18,
            # 19,
            # 20,
        ],  # x up to 20
        [1, 1, 2, 3, 4, 5, 6, 7, 8],  # x up to 10
        [1, 1, 2, 3, 4, 5, 6],  # x up to 5
    ]
    colors = ["blue", "red", "green"]
    markers = ["o", "s", "^"]
    # Set up the plot
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({"font.size": 16})
    plt.rcParams["savefig.directory"] = os.path.expanduser(
        "~/Programming/RobotMower/finalReport/images/"
    )
    # Run tests for each x array
    for idx, x in enumerate(x_arrays):
        max_x = max(x)
        print(f"\nRunning tests with x values up to {max_x}")
        runtime_usages = run_runtime_test(x)
        # Print results
        for xi, avg_runtime in zip(x, runtime_usages):
            print(f"x = {xi}: average runtime = {avg_runtime:.2f} ms")
        # Remove first duplicate entry (same as original code)
        x_plot = x[1:]  # Remove first element
        runtime_plot = runtime_usages[1:]  # Remove first element
        # Plot the data
        plt.scatter(
            x_plot,
            runtime_plot,
            color=colors[idx],
            marker=markers[idx],
            s=100,
            alpha=0.7,
            label=f"Number of runs {len(x) - 1}",
        )
    # Customize the plot
    plt.xlabel("X Values")
    plt.ylabel("Runtime (ms)")
    plt.title("Runtime Comparison")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
