import argparse
import os

from aerialMapping.runModel import run_model_and_get_outlines
from aerialMapping.utils import save_outlines_to_json

# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), 'aerialMapping'))
# from testWeights import genMap


def main():
    parser = argparse.ArgumentParser(description="Process an image file")
    parser.add_argument("image_path", help="Path to the image file")

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found")
        return

    img = args.image_path

    print("Welcome!")
    # print(f"Image {img} detected")
    outline = run_model_and_get_outlines(img)

    print(outline)

    path = save_outlines_to_json(outline, img)

    print(path)


if __name__ == "__main__":
    main()
