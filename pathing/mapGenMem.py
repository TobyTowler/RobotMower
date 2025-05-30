import statistics
import os
import tracemalloc
import Mapping as m
import matplotlib.pyplot as plt


def run_memory_test(x_values):
    """Run memory test for given x values"""
    memory_usages = []
    for i in x_values:
        thisMemory = []
        for j in range(5):  # Keep original 5 runs
            tracemalloc.start()
            hull = []  # move inside to avoid growing across tests
            origin = m.Point(2, 2)
            field = m.genPoints(10, origin, 400)  # Keep original parameters
            hull.append(m.sortPoints(field, origin))
            current, peak = tracemalloc.get_traced_memory()
            thisMemory.append(peak / 1024)  # KB
            tracemalloc.stop()
        memory_usages.append(statistics.mean(thisMemory))
    return memory_usages


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
        memory_usages = run_memory_test(x)

        # Print results
        for xi, avg_memory in zip(x, memory_usages):
            print(f"x = {xi}: average peak memory = {avg_memory:.2f} KB")

        # Remove first duplicate entry (same as original code)
        x_plot = x[1:]  # Remove first element
        memory_plot = memory_usages[1:]  # Remove first element

        # Plot the data
        plt.scatter(
            x_plot,
            memory_plot,
            color=colors[idx],
            marker=markers[idx],
            s=100,
            alpha=0.7,
            label=f"x up to {max_x}",
        )

        # Connect points with lines for better visualization
        # plt.plot(
        #     x_plot,
        #     memory_plot,
        #     color=colors[idx],
        #     alpha=0.5,
        #     linestyle="-",
        #     linewidth=1,
        # )
        #
        # # Customize the plot
        # plt.scatter(
        #     x_plot,
        #     memory_plot,
        #     color=colors[idx],
        #     marker=markers[idx],
        #     s=100,
        #     alpha=0.7,
        #     label=f"Parameter = {param}",
        # )

        # Connect points with lines for better visualization
        # plt.plot(
        #     x_plot,
        #     memory_plot,
        #     color=colors[idx],
        #     alpha=0.5,
        #     linestyle="-",
        #     linewidth=1,
        # )
        #
    # Customize the plot
    plt.xlabel("X Values")
    plt.ylabel("Peak Memory Usage (KB)")
    plt.title("Peak Memory Usage Comparison")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
