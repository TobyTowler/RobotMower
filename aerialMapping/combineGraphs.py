import matplotlib.pyplot as plt
import numpy as np

holes_memory_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
holes_memory_y = np.array([9.57, 9.57, 9.57, 9.57, 9.57, 9.57, 9.57, 9.57, 9.57])

runs_memory_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
runs_memory_y = np.array([9.53, 9.53, 9.53, 9.53, 9.53, 9.53, 9.53, 9.53, 9.53])

plt.figure(figsize=(12, 8))

plt.plot(
    holes_memory_x,
    holes_memory_y,
    "o-",
    color="blue",
    linewidth=2.5,
    markersize=8,
    label="Memory Usage vs Number of Holes",
    alpha=0.8,
)
plt.plot(
    runs_memory_x,
    runs_memory_y,
    "s-",
    color="red",
    linewidth=2.5,
    markersize=8,
    label="Memory Usage vs Number of Runs",
    alpha=0.8,
)

plt.xlabel("Parameter Value", fontsize=16)
plt.ylabel("Peak Memory Usage (KB)", fontsize=16)
plt.title("Memory Usage: Number of Holes vs Number of Runs", fontsize=18, pad=20)

plt.grid(True, linestyle="--", alpha=0.6)

plt.legend(fontsize=14, loc="upper right", frameon=True, fancybox=True, shadow=True)

plt.xlim(0.5, 10.5)
plt.ylim(9.52, 9.58)

plt.xticks(range(1, 11), fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()

plt.axhline(y=9.57, color="blue", linestyle=":", alpha=0.5, linewidth=1)
plt.axhline(y=9.53, color="red", linestyle=":", alpha=0.5, linewidth=1)


plt.show()
