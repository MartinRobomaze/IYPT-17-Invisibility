import numpy as np


class Ray:
    """
    Represents a ray of light. Stores current parameters and all refraction/reflection points.
    """
    def __init__(self, x0, y0, initial_slope=np.inf, initial_dir='R', color='red'):
        """
        Initializes a new ray

        :param x0: Initial x point
        :param y0: Initial y point
        :param initial_slope: Initial slope/steepness of ray
        :param initial_dir: Initial direction - 'U' for up, 'D for down'
        :param color: Color of ray to be displayed
        """
        self.x = x0
        self.y = y0
        self.slope = initial_slope
        self.offset = self.y - self.slope * self.x
        self.direction = initial_dir
        self.collisions = 0
        self.prev_dir = self.direction
        self.reflection = False
        self.reflection_origin = None
        self.color = color

        self.points = [(self.x, self.y)]

    def update(self, x, y, slope, direction, new_collision=True):
        """
        Updates the ray - used to add points of refraction/reflection.

        :param x: X coordinate of refraction/reflection point
        :param y: Y coordinate of refraction/reflection point
        :param slope: New slope of ray
        :param direction: New direction of ray
        :param new_collision: True if ray enters a different optical medium
        :return:
        """
        self.prev_dir = direction
        self.points.append((x, y))
        self.x = x
        self.y = y
        self.slope = slope
        self.offset = self.y - self.slope * self.x
        self.direction = direction

        if new_collision:
            self.collisions += 1

    def calc_x(self, y):
        """
        Calculates X coordinate of ray for a given Y coordinate.

        :param y:
        :return: X coordinate of ray at Y
        """
        return (self.slope * self.x + y - self.y) / self.slope
