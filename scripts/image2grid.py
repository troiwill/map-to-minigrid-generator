from __future__ import annotations

import click
import cv2
from minigrid_json.grid_json import GridJson
import os

from image2minigrid.grid_generator import Image2MinigridGenerator


@click.command()
@click.argument("imagepath")
@click.argument("width", type=int)
@click.argument("height", type=int)
@click.argument("savepath")
@click.option("--env_name", help="Name of the environment.")
def image2grid(imagepath: str, width: int, height: int, savepath: str, env_name: str) -> None:
    """Convert an image to a Minigrid environment and save it as JSON.

    Args:

        imagepath (str): The image to convert to a Minigrid.

        width (int): The width of the Minigrid environment.

        height (int): The height of the Minigrid environment.

        savepath (str): The path to save the output JSON file.

        env_name (str): Name of the environment.

    Returns:
        None

    Raises:
        AssertionError: If the image path is invalid or if width or height are not positive.
    """
    # Sanity checks.
    assert os.path.exists(imagepath) and (imagepath.endswith(".jpg") or imagepath.endswith(".png"))
    assert width > 0
    assert height > 0
    savedir, filename = os.path.split(savepath)

    # Create the grid using the Image to Minigrid generator.
    click.echo("Creating environment from image.")
    grid = Image2MinigridGenerator.convert_image_to_grid(
        image=cv2.imread(imagepath),
        width=width,
        height=height,
    )

    if env_name is None or env_name == "":
        env_name, _ = os.path.splitext(filename)
    if not os.path.exists(savedir):
        click.echo(f"Creating the save directory: {savedir}")
        os.makedirs(savedir)
    click.echo(f"Exporting a grid named '{env_name}' to JSON: {savepath}")
    GridJson.export(grid, savepath, env_name)


if __name__ == "__main__":
    image2grid()
