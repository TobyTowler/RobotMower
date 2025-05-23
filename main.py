import argparse
import os

from aerialMapping.testWeights import genMap


def main():
    parser = argparse.ArgumentParser(description="Process an image file")
    parser.add_argument("image_path", help="Path to the image file")

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found")
        return

    if args.verbose:
        print(f"Processing image: {args.image_path}")
        if args.output:
            print(f"Output will be saved to: {args.output}")

    img = args.image_path

    print("Welcome!")
    print(f"Image {img} detected")
    genMap(img)


if __name__ == "__main__":
    main()
