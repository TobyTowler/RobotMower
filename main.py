import argparse
import os

import fields2cover as F2C

from aerialMapping.runModel import run_model_and_get_outlines
from aerialMapping.utils import save_outlines_to_json

from pathing.roughToF2C import genRoughPath
from pathing.fairwayToF2C import genFairwayPath
from pathing.greenToF2C import genGreenPath
from pathing.utils import save_route_to_json

# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), 'aerialMapping'))
# from testWeights import genMap


def main():
    parser = argparse.ArgumentParser(description="Process an image file")
    parser.add_argument(
        "image_path",
        nargs="?",
        default="./aerialMapping/imgs/testingdata/Benniksgaard_Golf_Klub_1000_02_2.jpg",
        help="Path to the image file",
    )

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found")
        return

    img = args.image_path

    print("Welcome!")

    outline = run_model_and_get_outlines(img)

    print(outline)

    path = save_outlines_to_json(outline, img)

    print(path)

    filename = os.path.basename(path)
    print("Generating fairway paths")
    fairwayPath = genFairwayPath(path)

    save_route_to_json(fairwayPath, "outputs/paths/" + filename + "fairwayPath.json")

    print("Generating rough paths")
    roughPath = genRoughPath(path)
    save_route_to_json(roughPath, "outputs/paths/" + filename + "roughPath.json")

    print("Generating green paths")
    greenPath = genGreenPath(path)
    save_route_to_json(greenPath, "outputs/paths/" + filename + "greenPath.json")


if __name__ == "__main__":
    main()
