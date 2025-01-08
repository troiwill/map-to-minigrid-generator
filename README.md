# map-to-minigrid-generator
A tool that converts a screenshot of a Google Map to a Minigrid environment.

## Quick Usage

Use the `convert_image_to_grid` method to convert a Google Maps image into a Minigrid environment.
```
from image2minigrid.grid_generator import Image2MinigridGenerator

grid = Image2MinigridGenerator.convert_image_to_grid(
    image,              # A NumPy array in BGR format (loaded with cv2)
    width,              # The width of the environment (in Minigrid cells)
    height,             # The height of the environment (in Minigrid cells)
    goal_pos,           # The goal position within the Minigrid environment
    free_space_color,   # Color (BGR format) to determine which cells will be free space
    occupied_cell_obj,  # The Minigrid WorldObj instance used to represent occupied cells
    grid_fn,            # The Minigrid Grid class or a subclass used to create the internal Grid.
)
```

For more detailed usage, please look at the scripts in the `tests` directory.

## Development Installation

Install using the following instructions within a virtual environment. 
```
git clone https://github.com/troiwill/minigrid-json     # Clone the Minigrid-JSON
cd minigrid-json                                        # Go into the directory
pip install -e .                                        # Pip install the package
```

## Test Scripts

To run the test scripts, run the following code:
```
cd tests
python gen_minigrid_from_img.py
python gen_minigrid_from_json.py
```
