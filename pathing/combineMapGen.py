import matplotlib.pyplot as plt
import numpy as np

# Data from your graphs
# Graph 1: Number of Runs vs Execution Time
# runs_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# runs_x = [5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200]
runs_x = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
runs_y = [
    0.0918,
    0.0848,
    0.0820,
    0.0842,
    0.0818,
    0.0906,
    0.0783,
    0.0991,
    0.0849,
    0.0810,
]

# Graph 2: Number of Holes vs Execution Time
holes_x = [5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200]
holes_y = [0.2, 0.4, 0.8, 1.2, 1.9, 2.2, 3.4, 4.4, 5.2, 6.6, 7.4, 8.6]

# Graph 3: Number of Points vs Execution Time
points_x = [5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200]
points_y = [0.06, 0.09, 0.15, 0.24, 0.32, 0.37, 0.59, 0.72, 0.92, 1.08, 1.27, 1.44]

# Graph 4: Range of Points vs Execution Time
# range_x = [0, 200, 400, 600, 800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800]
# range_y = [
#     0.0871,
#     0.0802,
#     0.0819,
#     0.0816,
#     0.0828,
#     0.0825,
#     0.0825,
#     0.1025,
#     0.0893,
#     0.0813,
#     0.0794,
#     0.0819,
#     0.0860,
# ]
#
# Create the combined plot
plt.figure(figsize=(20, 16))

# Plot each dataset with different colors and markers
plt.scatter(
    runs_x,
    runs_y,
    color="blue",
    marker="o",
    s=100,
    alpha=0.7,
    label="Average of Baseline Runs",
)
plt.scatter(
    holes_x, holes_y, color="red", marker="s", s=100, alpha=0.7, label="Number of Holes"
)
plt.scatter(
    points_x,
    points_y,
    color="green",
    marker="^",
    s=100,
    alpha=0.7,
    label="Number of Points",
)
# plt.scatter(
#     range_x,
#     range_y,
#     color="orange",
#     marker="D",
#     s=100,
#     alpha=0.7,
#     label="Range of Points",
# )

# Optional: Add connecting lines for each dataset
plt.plot(runs_x, runs_y, color="blue", alpha=0.3, linestyle="-", linewidth=1)
plt.plot(holes_x, holes_y, color="red", alpha=0.3, linestyle="-", linewidth=1)
plt.plot(points_x, points_y, color="green", alpha=0.3, linestyle="-", linewidth=1)
# plt.plot(range_x, range_y, color="orange", alpha=0.3, linestyle="-", linewidth=1)

# Customize the plot
plt.xlabel("Parameter Value", fontsize=22)
plt.ylabel("Execution Time (milliseconds)", fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(loc="upper left", fontsize=20)
plt.grid(True, linestyle="--", alpha=0.7)

# Adjust layout and display
# plt.tight_layout()
plt.show()

# Optional: Save the plot
# plt.savefig('combined_performance_analysis.png', dpi=300, bbox_inches='tight')

# print("Plot generated successfully!")
# print("\nData Summary:")
# print(f"Number of Runs: {len(runs_x)} points, range {min(runs_x)}-{max(runs_x)}")
# # print(f"Number of Holes: {len(holes_x)} points, range {min(holes_x)}-{max(holes_x)}")
# print(
#     f"Number of Points: {len(points_x)} points, range {min(points_x)}-{max(points_x)}"
# )
# print(f"Range of Points: {len(range_x)} points, range {min(range_x)}-{max(range_x)}")
