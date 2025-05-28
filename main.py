import argparse
import os

import app

if __name__ == "__main__":
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
        # app.run("")
        # return

    img = args.image_path
    app.run(img)
