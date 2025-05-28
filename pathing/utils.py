import fields2cover as f2c
import csv
import json
from datetime import datetime


def genField(csv_points):
    ring = f2c.LinearRing()
    for p in csv_points:
        print(p)
        ring.addGeometry(f2c.Point(p[0], p[1]))

    cell = f2c.Cell()
    cell.addRing(ring)

    cells = f2c.Cells()
    cells.addGeometry(cell)

    return cells


def load_csv_points(path, reverse=False):
    points = []
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile)

        next(reader)
        for row in reader:
            if reverse:
                point_id, y, x = int(row[0]), float(row[1]), float(row[2])
            else:
                point_id, x, y = int(row[0]), float(row[1]), float(row[2])
            points.append((x, y))
    return points


def mowerConfig(width, bladeWidth):
    """All lengths in m"""

    mower = f2c.Robot(width, bladeWidth)
    mower.setMinTurningRadius(0.15)
    mower.setMaxDiffCurv(0.1)
    return mower


def save_points_to_csv(points, filename):
    with open("fields/" + filename + ".csv", "w") as file:
        file.write("point_id,x,y\n")

        for i, point in enumerate(points, 1):
            x, y = point
            file.write(f"{i},{x},{y}\n")

    print(f"Points successfully saved to {filename}")


def drawCell(arr):
    f2c.Visualizer.figure()
    for i in arr:
        f2c.Visualizer.plot(i)
    f2c.Visualizer.show()


def getRobotCoords():
    return [2, 4]


def save_route_to_json(route, filename):
    if route is None or route.isEmpty():
        print("No route to save")
        return

    print(f"Extracting COMPLETE route with length: {route.length():.2f} meters")

    all_waypoints = []

    try:
        num_swath_vectors = route.sizeVectorSwaths()
        num_connections = route.sizeConnections()

        print(
            f"Route has {num_swath_vectors} swath vectors and {num_connections} connections"
        )

        total_swaths = 0
        for swath_vector_idx in range(num_swath_vectors):
            try:
                swaths = route.getSwaths(swath_vector_idx)
                total_swaths += len(swaths)
                print(f"Swath vector {swath_vector_idx}: {len(swaths)} swaths")

                for swath_idx, swath in enumerate(swaths):
                    try:
                        linestring = swath.asLineString()

                        points_found = 0

                        if hasattr(linestring, "getNumPoints"):
                            try:
                                num_points = linestring.getNumPoints()
                                print(
                                    f"    Swath {swath_idx} has {num_points} points (getNumPoints)"
                                )
                                for i in range(num_points):
                                    point = linestring.getPoint(i)
                                    all_waypoints.append(
                                        {
                                            "index": len(all_waypoints),
                                            "swath_vector": swath_vector_idx,
                                            "swath_id": swath_idx,
                                            "point_in_swath": i,
                                            "type": "swath_detail",
                                            "x": point.getX(),
                                            "y": point.getY(),
                                        }
                                    )
                                    points_found += 1
                            except Exception as e:
                                print(f"      getNumPoints failed: {e}")

                        if points_found == 0 and hasattr(linestring, "size"):
                            try:
                                size = linestring.size()
                                print(f"    Swath {swath_idx} size: {size}")
                                for i in range(size):
                                    point = linestring[i]
                                    all_waypoints.append(
                                        {
                                            "index": len(all_waypoints),
                                            "swath_vector": swath_vector_idx,
                                            "swath_id": swath_idx,
                                            "point_in_swath": i,
                                            "type": "swath_detail",
                                            "x": point.getX(),
                                            "y": point.getY(),
                                        }
                                    )
                                    points_found += 1
                            except Exception as e:
                                print(f"      size() method failed: {e}")

                        if points_found == 0:
                            for method_name in [
                                "getCoords",
                                "getCoordinates",
                                "coords",
                                "coordinates",
                            ]:
                                if hasattr(linestring, method_name):
                                    try:
                                        coords = getattr(linestring, method_name)()
                                        print(
                                            f"    Using {method_name}() - found {len(coords)} coords"
                                        )
                                        for i, coord in enumerate(coords):
                                            all_waypoints.append(
                                                {
                                                    "index": len(all_waypoints),
                                                    "swath_vector": swath_vector_idx,
                                                    "swath_id": swath_idx,
                                                    "point_in_swath": i,
                                                    "type": "swath_detail",
                                                    "x": coord.getX(),
                                                    "y": coord.getY(),
                                                }
                                            )
                                            points_found += 1
                                        break
                                    except Exception as e:
                                        continue

                        if points_found == 0:
                            try:
                                for i, point in enumerate(linestring):
                                    all_waypoints.append(
                                        {
                                            "index": len(all_waypoints),
                                            "swath_vector": swath_vector_idx,
                                            "swath_id": swath_idx,
                                            "point_in_swath": i,
                                            "type": "swath_detail",
                                            "x": point.getX(),
                                            "y": point.getY(),
                                        }
                                    )
                                    points_found += 1
                                if points_found > 0:
                                    print(f"    Iteration found {points_found} points")
                            except Exception as e:
                                print(f"      Iteration failed: {e}")

                        if points_found == 0:
                            start = swath.startPoint()
                            end = swath.endPoint()

                            all_waypoints.extend(
                                [
                                    {
                                        "index": len(all_waypoints),
                                        "swath_vector": swath_vector_idx,
                                        "swath_id": swath_idx,
                                        "type": "swath_start",
                                        "x": start.getX(),
                                        "y": start.getY(),
                                    },
                                    {
                                        "index": len(all_waypoints),
                                        "swath_vector": swath_vector_idx,
                                        "swath_id": swath_idx,
                                        "type": "swath_end",
                                        "x": end.getX(),
                                        "y": end.getY(),
                                    },
                                ]
                            )
                            points_found = 2
                            print(f"    Used start/end points as fallback")

                        print(f"    Swath {swath_idx}: extracted {points_found} points")

                    except Exception as e:
                        print(f"    Error processing swath {swath_idx}: {e}")

            except Exception as e:
                print(f"  Error processing swath vector {swath_vector_idx}: {e}")

        print(f"Processed {total_swaths} total swaths")

        print(f"Processing ALL {num_connections} connections...")
        connections_processed = 0

        for conn_idx in range(num_connections):
            try:
                connection = route.getConnection(conn_idx)
                connection_points = 0

                if hasattr(connection, "getNumGeometries"):
                    try:
                        num_geoms = connection.getNumGeometries()
                        for i in range(num_geoms):
                            geom = connection.getGeometry(i)
                            all_waypoints.append(
                                {
                                    "index": len(all_waypoints),
                                    "connection_id": conn_idx,
                                    "point_in_connection": i,
                                    "type": "connection_point",
                                    "x": geom.getX(),
                                    "y": geom.getY(),
                                }
                            )
                            connection_points += 1
                    except:
                        pass

                if connection_points == 0 and hasattr(connection, "size"):
                    try:
                        size = connection.size()
                        for i in range(size):
                            point = connection[i]
                            all_waypoints.append(
                                {
                                    "index": len(all_waypoints),
                                    "connection_id": conn_idx,
                                    "point_in_connection": i,
                                    "type": "connection_point",
                                    "x": point.getX(),
                                    "y": point.getY(),
                                }
                            )
                            connection_points += 1
                    except:
                        pass

                if connection_points == 0:
                    try:
                        for i, point in enumerate(connection):
                            all_waypoints.append(
                                {
                                    "index": len(all_waypoints),
                                    "connection_id": conn_idx,
                                    "point_in_connection": i,
                                    "type": "connection_point",
                                    "x": point.getX(),
                                    "y": point.getY(),
                                }
                            )
                            connection_points += 1
                    except:
                        pass

                if connection_points > 0:
                    connections_processed += 1

                if conn_idx % 50 == 0:
                    print(f"  Processed {conn_idx}/{num_connections} connections...")

            except Exception as e:
                if conn_idx < 5:
                    print(f"  Error processing connection {conn_idx}: {e}")

        print(f"Successfully processed {connections_processed} connections")

    except Exception as e:
        print(f"Major error extracting route data: {e}")
        return

    if all_waypoints:
        route_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_waypoints": len(all_waypoints),
                "route_length_meters": route.length(),
                "num_swath_vectors": num_swath_vectors,
                "total_swaths": total_swaths,
                "num_connections": num_connections,
                "connections_processed": connections_processed,
            },
            "waypoints": all_waypoints,
        }

        with open(filename, "w") as f:
            json.dump(route_data, f, indent=2)

        print(f"route saved to {filename}")
        print(f"   - {len(all_waypoints)} total waypoints")
        print(f"   - {route.length():.2f} meters total length")
        print(
            f"   - {total_swaths} swaths, {connections_processed} connections processed"
        )
        return filename
    else:
        print("❌ No waypoints extracted from route")
        return None


def genPath(field, mower):
    const_hl = f2c.HG_Const_gen()
    no_hl = const_hl.generateHeadlands(field, 3.0 * mower.getWidth())
    n_swath = f2c.OBJ_NSwath()
    bf_sw_gen = f2c.SG_BruteForce()
    swaths_bf_nswath = bf_sw_gen.generateBestSwaths(
        n_swath, mower.getCovWidth(), no_hl.getGeometry(0)
    )
    boustrophedon_sorter = f2c.RP_Boustrophedon()
    swaths = boustrophedon_sorter.genSortedSwaths(swaths_bf_nswath)

    path_planner = f2c.PP_PathPlanning()
    dubins = f2c.PP_DubinsCurves()
    path_dubins = path_planner.planPath(mower, swaths, dubins)

    return [field, swaths, no_hl, path_dubins]
