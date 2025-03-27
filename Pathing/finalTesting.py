from utils import (
    load_csv_points,
    genField,
    drawCell,
    mowerConfig,
    save_points_to_csv,
    genPath,
)
from mapOrientation import rotateMap, scaleMap


def main():
    mower = mowerConfig(0.5, 0.3)

    path = "coords/garden.csv"
    gpsPath = "gps/gardenGoogleMaps.csv"

    gps = load_csv_points(gpsPath)
    points = load_csv_points(path)

    # field = genField(points)
    gpsField = genField(gps)

    # drawCell(field)
    drawCell(gpsField)

    rotatedPoints = rotateMap(points, gps)

    scaledPoints = scaleMap(rotatedPoints, gps)
    scaledField = genField(scaledPoints)

    scaledPath = genPath(scaledField, mower)
    drawCell(scaledPath)

    save_points_to_csv(scaledPoints, "transformedField")


if __name__ == "__main__":
    main()
