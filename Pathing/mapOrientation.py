import math
from utils import load_csv_points, genField, drawCell, save_points_to_csv
import geopy.distance

"""
get shape
get gps

sheer shape to gps
rotate shape
scale shape
transform to gps coordinates


"""


def rotateMap(points, gps):
    """
    Bearing in radians
    """

    if len(gps) < 2:
        raise ValueError("Need at least 2 GPS coordinates to calculate bearing")

    lat1, lon1 = gps[0]
    lat2, lon2 = gps[1]

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # θ = atan2(sin(Δlong) * cos(lat2), cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(Δlong))
    delta_lon = lon2_rad - lon1_rad
    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(
        lat2_rad
    ) * math.cos(delta_lon)

    bearing_rad = math.atan2(x, y)

    rotation_angle = -bearing_rad  # Negative because of different coordinate systems

    cos_angle = math.cos(rotation_angle)
    sin_angle = math.sin(rotation_angle)

    rotated_points = []
    for point in points:
        x, y = point

        rotated_x = x * cos_angle - y * sin_angle
        rotated_y = x * sin_angle + y * cos_angle
        rotated_points.append((rotated_x, rotated_y))

    return rotated_points


def sheerMap(map):
    pass


def translateMap(map):
    pass


def getLineLength(map: list[float]):
    """
    map : array of 2 2d points
    returns displacement between them
    """
    print(map)
    x1, y1, x2, y2 = map
    x = abs(x1 - x2)
    y = abs(y1 - y2)
    return math.sqrt(x**2 + y**2)


def scaleMap(map, gps):
    """
    scale = gps line length / map line length

    have x coords
    get distance
    scale everything by that

    or
    ratio points
    """

    gpsList = gps[:2]
    gpsMeters = geopy.distance.geodesic(gpsList[0], gpsList[1]).m

    # gpsLineLength = getLineLength([gps[0:2]])
    mapLineLength = getLineLength([map[0:2]])

    scale = gpsMeters / mapLineLength
    print(scale)
    scaledMap = []

    pass


def main():
    path = "coords/garden2.csv"
    gpsPath = "gps/garden.csv"
    # path = "coords/AdobeGold_golf_course_outline.csv"
    gps = load_csv_points(gpsPath, reverse=True)
    points = load_csv_points(path)

    scaleMap(points, gps)
    field = genField(points)
    gpsField = genField(gps)
    # drawCell(field)
    # drawCell(gpsField)

    newPoints = rotateMap(points, gps)

    # drawCell(genField(newPoints))

    save_points_to_csv(newPoints, "rotatedField")


if __name__ == "__main__":
    main()
