# Map generation and path planning of an Agri Robot

## What it is
A machine learning powered system to generated paths for mowing golf courses with autonomous robots.

Generates 3 types of paths for different sections of the golf course
- Rough
- Fairway
- Green

Model is specialised on Danish golf courses using orthophotos from the [Danish Golf Course Dataset](https://universe.roboflow.com/sportedge/danish-golf).

Outlines stored in `outputs/`
- map outlines
- GPS transformed map outlines
- paths

### Requirements
- [Fields2Cover](https://github.com/Fields2Cover/Fields2Cover)
- Python (built on 3.13 should work with others idk)


## How to use
For GUI (recommended):

```bash
python guimain.py
```

- Use buttons to select image
    * Can be found in `aerialMapping/imgs/rawImgs`
- Chose GPS translation or not
    * GPS coordinates need to input to corresponding corners
    * use Google Maps if not in Denmark :)

For TUI:
```bash
python main.py [OPTIONS]
```

Options:
    --pathToImage

- Default image is provided if path is not specified.
- No GPS support
