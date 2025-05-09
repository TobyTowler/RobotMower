import time
import os
import matplotlib.pyplot as plt
import statistics
import Mapping as m


def main():
    # x = [5, 10, 20, 30, 40, 50, 60, 80, 100, 125, 150, 175, 200]
    # x = [50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]
    # x = [1, 5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200]
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    times = []

    hull = []
    for i in x:
        thisTimes = []
        for j in range(5):
            startTime = time.perf_counter()
            origin = m.Point(2, 2)
            # field = m.genPoints(10, origin, i)  # Range
            # field = m.genPoints(i, origin, 400)  # Points
            field = m.genPoints(10, origin, 400)
            hull.append(m.sortPoints(field, origin))

            # for y in range(i):  # holes
            #     hole1Base = m.Point(100, 100)
            #     hole1Points = m.genPoints(5, hole1Base, 50)
            #     hull.append(m.sortPoints(hole1Points, hole1Base))

            endTime = time.perf_counter()
            thisTimes.append((endTime - startTime) * 1000)

        times.append(statistics.mean(thisTimes))

    plt.figure(figsize=(10, 10))
    plt.rcParams.update({"font.size": 16})
    plt.rcParams["savefig.directory"] = os.path.expanduser(
        "~/Programming/RobotMower/finalReport/images/"
    )
    plt.scatter(x, times, color="blue", marker="o", s=100, alpha=0.7)

    # plt.xlabel("Range on points")
    plt.xlabel("Number of Runs")
    plt.ylabel("Execution Time (milliseconds)")
    # plt.title("Performance of map generation algorithm baseline with 3 holes")
    # plt.title("Map Generation Runtime vs Number of Holes")

    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
