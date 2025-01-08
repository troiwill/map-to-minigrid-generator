from __future__ import annotations

import cv2
from minigrid.manual_control import ManualControl
from minigrid_json.grid_json import GridJson
import os

from image2minigrid.grid_generator import Image2MinigridGenerator

from stt_harbor_env import STTHarborEnv


if __name__ == "__main__":
    # Generate the Minigrid from a Google Map image.
    image = cv2.imread("../env_images/st_thomas_no_labels.png")
    grid = Image2MinigridGenerator.convert_image_to_grid(
        image=image,
        width=STTHarborEnv.WIDTH,
        height=STTHarborEnv.HEIGHT,
        goal_pos=STTHarborEnv.GOAL_POS,
    )

    JSON_FILE = "stt_waterfront.json"
    if os.path.exists(JSON_FILE):
        os.remove(JSON_FILE)

    # Export the grid to disk.
    GridJson.export(grid, JSON_FILE, "St. Thomas Waterfront")

    # Create the Minigrid environment. The _gen_grid function loads the
    env = STTHarborEnv(
        agent_start_pos=STTHarborEnv.START_POS,
        goal_pos=STTHarborEnv.GOAL_POS,
        json_file=JSON_FILE,
        render_mode="human",
        width=STTHarborEnv.WIDTH,
        height=STTHarborEnv.HEIGHT,
    )
    manual_control = ManualControl(env, seed=42)
    manual_control.start()
