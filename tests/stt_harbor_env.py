from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from minigrid_json.grid_json import GridJson
import numpy as np

from image2minigrid.grid_generator import Image2MinigridGenerator


class STTHarborEnv(MiniGridEnv):
    WIDTH: int = 120
    HEIGHT: int = 80
    START_POS: tuple[int, int] = (35, 50)
    GOAL_POS: tuple[int, int] = (40, 50)

    def __init__(
        self,
        goal_pos: tuple[int, int],
        agent_start_pos: tuple[int, int],
        agent_start_dir: int = 0,
        env_img: np.ndarray | None = None,
        json_file: str | None = None,
        width: int = 80,
        height: int = 60,
        max_steps: int = 1000,
        **kwargs,
    ):
        assert (isinstance(env_img, np.ndarray) and json_file is None) or (
            env_img is None and isinstance(json_file, str)
        )

        self._env_img = env_img
        self._json_file = json_file
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = goal_pos

        super().__init__(
            mission_space=MissionSpace(mission_func=self._gen_mission),
            width=width,
            height=height,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "get to the goal safely"

    def _gen_grid(self, width: int, height: int) -> None:
        if self._env_img is not None:
            # Create the grid using the Image to Minigrid generator.
            print("Creating environment from image.")
            self.grid = Image2MinigridGenerator.convert_image_to_grid(
                image=self._env_img,
                width=width,
                height=height,
                goal_pos=self.goal_pos,
            )

        else:
            # Loads the Minigrid from the JSON file.
            print(f"Creating environment from GridJSON {self._json_file}")
            self.grid = GridJson.load(self._json_file, Grid)

        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
