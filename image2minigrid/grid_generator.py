from __future__ import annotations
import copy
from typing import Tuple, Callable

from minigrid.core.grid import Grid
import numpy as np
import skimage.measure

from minigrid.core.world_object import Goal, Lava, WorldObj


class Image2MinigridGenerator:

    @staticmethod
    def convert_image_to_grid(
        image: np.ndarray,
        width: int,
        height: int,
        goal_pos: Tuple[int, int] | None = None,
        free_space_color: Tuple[int, int, int] = (255, 181, 148),
        occupied_cell_obj: WorldObj = Lava(),
        grid_fn: Callable[[int, int], Grid] = Grid,
    ) -> Grid:
        """
        Converts an image to a Minigrid Grid object.

        This method processes an input image and converts it into a Grid object for use in Minigrid environments.
        It identifies areas of 'land' based on a specified color threshold.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            width (int): The desired width of the output grid.
            height (int): The desired height of the output grid.
            goal_pos (Tuple[int, int] | None, optional): The position of the goal in the grid. Defaults to None.
            free_space_color (Tuple[int, int, int], optional): The RGB color representing free space in the image.
                                                               Defaults to (255, 181, 148).
            occupied_cell_obj (WorldObj, optional): The object to use in occupied cells.
            grid_fn (Callable[[int, int], Grid], optional): A function to create the grid. Defaults to Grid.

        Returns:
            Grid: A Minigrid Grid object representing the processed image.

        Raises:
            ValueError: If the image is not a NumPy array, or if height or width are not positive integers.

        Note:
            - The method uses min pooling to downscale the image to the desired grid dimensions.
            - Areas not matching the free_space_color are converted to Lava in the grid.
            - The grid is surrounded by walls.
            - If a goal_pos is provided, a Goal object is placed at that position in the grid.
        """

        if not isinstance(image, np.ndarray):
            raise ValueError(f"image must be an NDArray. Got type {type(image)}.")
        height, width = int(height), int(width)
        if height <= 0:
            raise ValueError("height must be a positive integer.")
        if width <= 0:
            raise ValueError("width must be a positive integer.")

        # Blue Channel Threshold to be consider water
        threshold_color = np.array(free_space_color)

        # Trim image to fit make it a multiple of x_dim and y_dim
        img = image.copy()
        img = img[
            : (img.shape[0] // height) * height, : (img.shape[1] // width) * width
        ]

        # Discrimate pixels of our desired color
        img = (
            (img[:, :, 0] == threshold_color[0])
            & (img[:, :, 1] == threshold_color[1])
            & (img[:, :, 2] == threshold_color[2])
        )
        img = (img).astype(np.uint8)
        img = img * 255

        # Min Pooling to downscale image
        img = skimage.measure.block_reduce(
            img, (int(img.shape[0] / height), int(img.shape[1] / width)), np.min
        )

        coordinates_of_land = np.argwhere(img == 0)
        list_land = list(map(tuple, coordinates_of_land))
        grid = grid_fn(width, height)
        grid.wall_rect(0, 0, width, height)
        for tup in list_land:
            # Ignore if it already on the border
            if (
                tup[0] != 0
                and tup[0] != height - 1
                and tup[1] != 0
                and tup[1] != width - 1
            ):
                grid.set(tup[1], tup[0], copy.deepcopy(occupied_cell_obj))

        if goal_pos is not None:
            grid.set(goal_pos[0], goal_pos[1], Goal())

        return grid
