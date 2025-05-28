import json
import matplotlib.pyplot as plt


def simple_route_plot(json_file):
    """Simple route visualization"""

    # Check if file exists first
    import os

    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        return

    with open(json_file, "r") as f:
        data = json.load(f)

    waypoints = data["waypoints"]

    x = [p["x"] for p in waypoints]
    y = [p["y"] for p in waypoints]

    plt.figure(figsize=(12, 8))
    plt.plot(x, y, "b-", linewidth=0.5, alpha=0.8)
    plt.scatter(x[0], y[0], c="green", s=100, label="Start")
    plt.scatter(x[-1], y[-1], c="red", s=100, label="End")

    plt.title(f"Mowing Route ({len(waypoints)} waypoints)")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.show()

    print(
        f"Route: {len(waypoints)} points, {data['metadata']['route_length_meters']:.2f}m total"
    )


# Fix the syntax error and use the correct path
if __name__ == "__main__":
    # Use the correct file path
    json_file = "../outputs/paths/Benniksgaard_Golf_Klub_1000_02_2_outlines.jsonfairwayPath.json"
    simple_route_plot(json_file)
