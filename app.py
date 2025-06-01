import os
import threading

import fields2cover as F2C

from aerialMapping.runModel import run_model_and_get_outlines
from aerialMapping.utils import save_outlines_to_json

from pathing.roughToF2C import genRoughPath
from pathing.fairwayToF2C import genFairwayPath
from pathing.greenToF2C import genGreenPath
from pathing.utils import load_csv_points, save_route_to_json
from pathing.mapTransform2 import transformPoints


def run(img, gui_instance, existing_outline_path=None):
    threading.Thread(
        target=process_with_progress,
        args=(img, gui_instance, existing_outline_path),
        daemon=True,
    ).start()


def process_with_progress(img, gui, existing_outline_path=None):
    print("Continuing processing...")
    filename = os.path.basename(img)

    # Use existing outline path if provided (from scan step)
    if existing_outline_path:
        path = existing_outline_path
        gui.updateProgress(30)
    else:
        # Original workflow - run model and save outlines
        gui.updateStep("Running model")
        outline = run_model_and_get_outlines(img)
        gui.updateProgress(20)

        print(outline)

        gui.updateProgress(25)
        path = save_outlines_to_json(outline, img)
        gui.updateStep("Saved outlines")
        gui.updateProgress(30)

    # GPS transformation if enabled
    if gui.useGPS:
        gui.updateStep("Transforming with GPS")
        gpsCoords = [gui.gps1.get(), gui.gps2.get()]

        # Validate GPS coordinates
        if not gpsCoords[0] or not gpsCoords[1]:
            gui.updateStep("Error: Please enter GPS coordinates")
            return

        path = transformPoints(path, gpsCoords)
        gui.updateProgress(35)

    print(f"Using path: {path}")

    print("Generating fairway paths")
    gui.updateStep("Generating fairway paths")
    gui.updateProgress(40)
    try:
        fairwayPath = genFairwayPath(path)
        gui.updateProgress(50)
        save_route_to_json(
            fairwayPath, "outputs/paths/" + filename + "fairwayPath.json"
        )
        gui.updateProgress(55)
    except Exception as e:
        print(f"Error generating fairway paths: {e}")
        gui.updateStep(f"Fairway path error: {e}")

    print("Generating rough paths")
    gui.updateStep("Generating rough paths")
    gui.updateProgress(60)
    try:
        roughPath = genRoughPath(path)
        gui.updateProgress(75)
        save_route_to_json(roughPath, "outputs/paths/" + filename + "roughPath.json")
        gui.updateProgress(80)
    except Exception as e:
        print(f"Error generating rough paths: {e}")
        gui.updateStep(f"Rough path error: {e}")

    print("Generating green paths")
    gui.updateStep("Generating green paths")
    gui.updateProgress(85)
    try:
        greenPath = genGreenPath(path)
        gui.updateProgress(95)
        save_route_to_json(greenPath, "outputs/paths/" + filename + "greenPath.json")
        gui.updateProgress(100)
    except Exception as e:
        print(f"Error generating green paths: {e}")
        gui.updateStep(f"Green path error: {e}")

    gui.updateStep("Process complete")
    print("Processing complete!")
