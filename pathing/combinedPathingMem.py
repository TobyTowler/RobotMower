import matplotlib.pyplot as plt
import numpy as np

# Data from Graph 1: Number of Runs vs Peak Memory Usage
runs_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
runs_y = [9.54, 9.54, 9.54, 9.54, 9.54, 9.54, 9.54, 9.54, 9.54, 9.54]

# Data from Graph 2: Number of Holes vs Peak Memory Usage
holes_x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
holes_y = [9.57, 9.57, 9.57, 9.57, 9.57, 9.57, 9.57, 9.57, 9.57]

# Data from Graph 3: Route Planning Methods vs Peak Memory Usage
methods = ["Shortest Route", "Boustrophedon Order", "Snake Order", "Spiral Order"]
methods_y = [7.80, 8.00, 8.00, 8.02]
methods_x = [1, 2, 3, 4]  # Position them at x = 1,2,3,4
method_colors = ["red", "orange", "green", "purple"]

# Data from Graph 4: Area of Field vs Peak Memory Usage
area_x = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
area_y = [9.536, 9.531, 9.536, 9.531, 9.557, 9.557, 9.557]

# Create the combined plot
plt.figure(figsize=(20, 16))

# Plot Number of Runs (blue circles)
plt.scatter(
    runs_x, runs_y, color="blue", marker="o", s=150, alpha=0.7, label="Number of Runs"
)

# Plot Number of Holes (cyan squares)
plt.scatter(
    holes_x,
    holes_y,
    color="cyan",
    marker="s",
    s=150,
    alpha=0.7,
    label="Number of Holes",
)

# Plot Route Planning Methods (different colors, diamonds)
for i, (x, y, color, method) in enumerate(
    zip(methods_x, methods_y, method_colors, methods)
):
    if i == 0:  # Only add label for first method to avoid cluttering legend
        plt.scatter(
            x,
            y,
            color=color,
            marker="D",
            s=200,
            alpha=0.8,
            label="Route Planning Methods",
        )
    else:
        plt.scatter(x, y, color=color, marker="D", s=200, alpha=0.8)

# Plot Area of Field (brown triangles)
plt.scatter(
    area_x,
    area_y,
    color="brown",
    marker="^",
    s=150,
    alpha=0.7,
    label="Area of Field (10^x)m^2",
)

# Add connecting lines
plt.plot(runs_x, runs_y, color="blue", alpha=0.3, linestyle="-", linewidth=2)
plt.plot(holes_x, holes_y, color="cyan", alpha=0.3, linestyle="-", linewidth=2)
plt.plot(area_x, area_y, color="brown", alpha=0.3, linestyle="-", linewidth=2)

# Add annotations for route planning methods
for i, (x, y, method) in enumerate(zip(methods_x, methods_y, methods)):
    plt.annotate(
        f"{method}\n{y:.2f} KB",
        xy=(x, y),
        xytext=(0, 20),
        textcoords="offset points",
        fontsize=12,
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

# Customize the plot
plt.xlabel("Parameter Value", fontsize=22)
plt.ylabel("Peak Memory Usage (KB)", fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=22)
plt.legend(loc="lower right", fontsize=18)
plt.grid(True, linestyle="--", alpha=0.7)

# Set y-axis limits to show the differences better
plt.ylim(7.5, 10.0)

# Fix the text cutoff issue
# plt.subplots_adjust(left=0.12, bottom=0.12, right=0.95, top=0.95)

plt.show()

# Optional: Save the plot
# plt.savefig('combined_memory_usage_analysis.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
