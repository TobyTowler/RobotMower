import os
import threading

import fields2cover as F2C

from aerialMapping.runModel import run_model_and_get_outlines
from aerialMapping.utils import save_outlines_to_json

from pathing.roughToF2C import genRoughPath
from pathing.fairwayToF2C import genFairwayPath
from pathing.greenToF2C import genGreenPath
from pathing.utils import load_csv_points, save_route_to_json


def run(img, gui_instance):
    threading.Thread(
        target=process_with_progress, args=(img, gui_instance), daemon=True
    ).start()


def process_with_progress(img, gui):
    print("Welcome!")
    gui.updateStep("Welcome")
    filename = os.path.basename(img)

    gui.updateProgress(5)
    gui.updateStep("Running model")
    outline = run_model_and_get_outlines(img)
    gui.updateProgress(20)

    print(outline)

    gui.updateProgress(25)
    gui.updateStep("Saved outlines")
    if gui.useGPS:
        path = load_csv_points("./outputs/transformedOutlines/" + filename + ".csv")

    else:
        path = save_outlines_to_json(outline, img)
    gui.updateProgress(30)

    print(path)

    print("Generating fairway paths")
    gui.updateStep("Generating fairway paths")
    gui.updateProgress(35)
    fairwayPath = genFairwayPath(path)
    gui.updateProgress(45)
    save_route_to_json(fairwayPath, "outputs/paths/" + filename + "fairwayPath.json")
    gui.updateProgress(50)

    print("Generating rough paths")
    gui.updateStep("Generating rough paths")
    gui.updateProgress(55)
    roughPath = genRoughPath(path)
    gui.updateProgress(70)
    save_route_to_json(roughPath, "outputs/paths/" + filename + "roughPath.json")
    gui.updateProgress(75)

    print("Generating green paths")
    gui.updateStep("Generating green paths")
    gui.updateProgress(80)
    greenPath = genGreenPath(path)
    gui.updateProgress(95)
    save_route_to_json(greenPath, "outputs/paths/" + filename + "greenPath.json")
    gui.updateProgress(100)

    gui.updateStep("Process complete")
    print("Processing complete!")
