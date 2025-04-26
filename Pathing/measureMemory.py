import statistics
import tracemalloc
import Mapping as m
import matplotlib.pyplot as plt


def main():
    # x = [1, 5, 10, 20, 30, 40, 50, 75, 100, 120, 150, 200]
    # x = [5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200]
    x = [50, 200, 400, 600, 800, 1000, 1200, 1400]
    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    memory_usages = []

    for i in x:
        thisMemory = []
        for j in range(5):
            tracemalloc.start()

            hull = []  # move inside to avoid growing across tests
            origin = m.Point(2, 2)
            field = m.genPoints(10, origin, i)
            hull.append(m.sortPoints(field, origin))

            # for y in range(i):  # holes
            #     hole1Base = m.Point(100, 100)
            #     hole1Points = m.genPoints(5, hole1Base, 50)
            #     hull.append(m.sortPoints(hole1Points, hole1Base))

            current, peak = tracemalloc.get_traced_memory()
            thisMemory.append(peak / 1024)  # KB
            tracemalloc.stop()

        memory_usages.append(statistics.mean(thisMemory))

    # Print results
    for xi, avg_memory in zip(x, memory_usages):
        print(f"x = {xi}: average peak memory = {avg_memory:.2f} KB")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(x, memory_usages, color="blue", marker="o", s=100, alpha=0.7)

    plt.xlabel("Range of Points")
    plt.ylabel("Peak Memory Usage (KB)")
    # plt.title("Peak Memory Usage vs Number of Holes in Map Generation")
    plt.title("Map Generation Peak Memory Usage vs Range of Points")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
