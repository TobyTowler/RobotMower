import matplotlib.pyplot as plt
import numpy as np

# Data from Graph 1: Number of Holes vs Peak Memory Usage
holes_x = [5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200]
holes_y = [3, 8, 15, 23, 31, 39, 58, 78, 100, 121, 142, 162]

# Data from Graph 2: Number of Points vs Peak Memory Usage
points_x = [5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200]
points_y = [1.5, 2, 3, 4.5, 6, 7.5, 11, 13.5, 17.5, 21, 26, 29]

# Create the combined plot
plt.figure(figsize=(20, 16))

# Plot each dataset with different colors and markers
plt.scatter(
    holes_x,
    holes_y,
    color="blue",
    marker="o",
    s=150,
    alpha=0.7,
    label="Number of Holes",
)

plt.scatter(
    points_x,
    points_y,
    color="red",
    marker="s",
    s=150,
    alpha=0.7,
    label="Number of Points",
)

# Optional: Add connecting lines for each dataset
plt.plot(holes_x, holes_y, color="blue", alpha=0.3, linestyle="-", linewidth=2)
plt.plot(points_x, points_y, color="red", alpha=0.3, linestyle="-", linewidth=2)

# Customize the plot
plt.xlabel("Parameter Value", fontsize=22)
plt.ylabel("Peak Memory Usage (KB)", fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(loc="upper left", fontsize=20)
plt.grid(True, linestyle="--", alpha=0.7)

# Fix the text cutoff issue
plt.subplots_adjust(left=0.12, bottom=0.12, right=0.95, top=0.95)

plt.show()

# Optional: Save the plot
# plt.savefig('merged_holes_points_memory_analysis.png', dpi=300, bbox_inches='tight', pad_inches=0.2)

print("Merged plot generated successfully!")
print(f"\nData Summary:")
print(f"Number of Holes: {len(holes_x)} points, range {min(holes_x)}-{max(holes_x)}")
print(
    f"Number of Points: {len(points_x)} points, range {min(points_x)}-{max(points_x)}"
)
print(f"Memory usage ranges:")
print(f"  Holes: {min(holes_y):.1f} - {max(holes_y):.1f} KB")
print(f"  Points: {min(points_y):.1f} - {max(points_y):.1f} KB")
print(f"\nGrowth rates:")
print(
    f"  Holes: ~{(max(holes_y) - min(holes_y)) / (max(holes_x) - min(holes_x)):.2f} KB per hole"
)
print(
    f"  Points: ~{(max(points_y) - min(points_y)) / (max(points_x) - min(points_x)):.2f} KB per point"
)
