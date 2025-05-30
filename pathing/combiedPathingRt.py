import matplotlib.pyplot as plt
import numpy as np

# Data from Graph 1: Number of Holes vs Execution Time
holes_x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
holes_y = [900, 1200, 2000, 2400, 2800, 3300, 3650, 4150, 4600, 5200]

# Data from Graph 2: Path Planning Methods vs Execution Time
methods = ["Shortest Route", "Boustrophedon Order", "Snake Order", "Spiral Order"]
methods_y = [6846.90, 31.88, 28.53, 36.72]
# Create x positions for the bar chart methods (we'll overlap with holes data)
methods_x = [0, 1, 2, 3]  # Use same x positions as first 4 hole values
# Different colors for each method
method_colors = ["red", "orange", "green", "purple"]

# Create the combined plot
plt.figure(figsize=(20, 16))

# Plot the holes data as scatter plot
plt.scatter(
    holes_x,
    holes_y,
    color="blue",
    marker="o",
    s=150,
    alpha=0.7,
    label="Number of Holes",
)

# Plot the path planning methods as scatter plot with different colors
for i, (x, y, color, method) in enumerate(
    zip(methods_x, methods_y, method_colors, methods)
):
    plt.scatter(x, y, color=color, marker="s", s=200, alpha=0.7, label=method)

# Add connecting line for holes data
plt.plot(holes_x, holes_y, color="blue", alpha=0.3, linestyle="-", linewidth=2)

# Customize the plot
plt.xlabel("Parameter / Method", fontsize=22)
plt.ylabel("Execution Time (milliseconds)", fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=22)

# Create custom x-tick labels - just use 0-9 for holes
plt.xticks(holes_x, [str(x) for x in holes_x])

plt.legend(loc="lower right", fontsize=20)
plt.grid(True, linestyle="--", alpha=0.7)

# Add annotations for the path planning methods with their values (under the points)
for i, (x, y, method) in enumerate(zip(methods_x, methods_y, methods)):
    plt.annotate(
        f"{y:.1f} ms",
        xy=(x, y),
        xytext=(0, -10),  # Place below the point
        textcoords="offset points",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="top",
        color="black",
    )

# Fix the text cutoff issue
# plt.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.95)

plt.show()

# Optional: Save the plot
# plt.savefig('merged_holes_pathplanning_analysis.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
