import os

import fields2cover as F2C

from aerialMapping.runModel import run_model_and_get_outlines
from aerialMapping.utils import save_outlines_to_json

from pathing.roughToF2C import genRoughPath
from pathing.fairwayToF2C import genFairwayPath
from pathing.greenToF2C import genGreenPath
from pathing.utils import save_route_to_json


def run(img):
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
