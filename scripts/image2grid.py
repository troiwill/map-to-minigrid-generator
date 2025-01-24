from __future__ import annotations

import click
import cv2
from minigrid.core.grid import Grid
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.manual_control import ManualControl
from minigrid_json.grid_json import GridJson
import os

from image2minigrid.grid_generator import Image2MinigridGenerator


class SimpleEnvironment(MiniGridEnv):

    def __init__(
        self,
        jsonfile,
        max_steps=100,
        see_through_walls=False,
        agent_view_size=7,
        render_mode=None,
        screen_size=640,
        highlight=True,
        agent_pov=False,
    ):
        self._env_desc = GridJson.load_description(jsonfile)
        self.agent_start_pos = (2, 2)
        self.agent_start_dir = 0

        super().__init__(
            mission_space=MissionSpace(mission_func=self._gen_mission),
            width=self._env_desc.width,
            height=self._env_desc.height,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            agent_view_size=agent_view_size,
            render_mode=render_mode,
            screen_size=screen_size,
            highlight=highlight,
            agent_pov=agent_pov,
        )

    def _gen_grid(self, width, height):
        self.grid = GridJson.convert_description_to_grid(self._env_desc, Grid)
        self.place_agent()

    @staticmethod
    def _gen_mission():
        return "visualizing simple environment."


@click.group()
def image2grid():
    """
    A utility for converting an image to a Minigrid environment or visualizing the map.
    """
    pass


@image2grid.command()
@click.argument("imagepath")
@click.argument("width", type=int)
@click.argument("height", type=int)
@click.argument("savedir")
@click.option("--env_name", help="Name of the environment.")
def convert(
    imagepath: str, width: int, height: int, savedir: str, env_name: str
) -> None:
    """Convert an image to a Minigrid environment and save it as JSON.

    Args:
        imagepath: The image to convert to a Minigrid.
        width: The width of the Minigrid environment.
        height: The height of the Minigrid environment.
        savedir: The directory to save the output JSON file.
        env_name: Name of the environment.
    """
    # Sanity checks.
    assert os.path.exists(imagepath) and (
        imagepath.endswith(".jpg") or imagepath.endswith(".png")
    )
    assert width > 0
    assert height > 0

    # Create the grid using the Image to Minigrid generator.
    click.echo("Creating environment from image.")
    grid = Image2MinigridGenerator.convert_image_to_grid(
        image=cv2.imread(imagepath),
        width=width,
        height=height,
    )

    if env_name is None or env_name == "":
        _, image_name = os.path.split(imagepath)
        env_name, _ = os.path.splitext(image_name)

    filename = f"{env_name}.json"
    savepath = os.path.join(savedir, filename)
    if env_name is None or env_name == "":
        env_name, _ = os.path.splitext(filename)
    if not os.path.exists(savedir):
        click.echo(f"Creating the save directory: {savedir}")
        os.makedirs(savedir)
    click.echo(f"Exporting a grid named '{env_name}' to JSON: {savepath}")
    GridJson.export(grid, savepath, env_name)


@image2grid.command()
@click.argument("jsonpath")
def show(jsonpath: str) -> None:
    """Shows the Minigrid environment loaded from the JSON file.

    Args:
        jsonpath: The JSON file to load into a Minigrid.
    """
    assert os.path.exists(jsonpath) and jsonpath.endswith(".json")
    env = SimpleEnvironment(
        jsonfile=jsonpath,
        render_mode="human",
    )
    manual_control = ManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    image2grid()
