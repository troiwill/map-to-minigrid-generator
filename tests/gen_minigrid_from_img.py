from __future__ import annotations

import cv2
from minigrid.manual_control import ManualControl

from stt_harbor_env import STTHarborEnv


if __name__ == "__main__":
    env = STTHarborEnv(
        cv2.imread("../env_images/st_thomas_no_labels.png"),
        agent_start_pos=STTHarborEnv.START_POS,
        goal_pos=STTHarborEnv.GOAL_POS,
        width=STTHarborEnv.WIDTH,
        height=STTHarborEnv.HEIGHT,
        render_mode="human",
    )
    manual_control = ManualControl(env, seed=42)
    manual_control.start()
