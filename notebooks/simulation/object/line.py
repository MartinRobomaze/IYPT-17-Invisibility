import numpy as np
from .iface import Object
from ..utils.math import derivative
from ..ray import Ray


class Line(Object):
    """
    Represents a line with a given length. Used to simulate planes
    """
    def __init__(self, x0, x1, y0, y1):
        """
        Initializes a new line.

        :param x0: X coordinate of starting point.
        :param x1: X coordinate of ending point.
        :param y0: Y coordinate of starting point.
        :param y1: Y coordinate of ending point.
        """
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1

        self.slope = (self.y1 - self.y0) / (self.x1 - self.x0)
        self.offset = self.y0 - self.slope * self.x0

    def y_coords(self, x):
        """
        Calculates the Y coordinate from a given X coordinate.

        :param x:
        :return:
        """
        return self.slope * x + self.offset

    def normal_line_slope(self, x):
        """
        Calculates the slope of a normal line to the line at point X.

        :param x: X coordinate of the normal line
        :return: Slope of the normal line
        """
        return np.inf if self.y0 - self.y1 == 0 else -1 / derivative(self.y_coords, x)

    def collides_with(self, ray: Ray):
        """
        Calculates the intersection points of a ray & a line.

        :param ray:
        :return: Intersection points of ray and circle, None if there are none
        """
        if ray.slope is np.inf:
            if self.x0 <= ray.x <= self.x1:
                return [(ray.x, self.y_coords(ray.x))]
            return None

        col_x = (ray.slope*ray.x - ray.y + self.offset) / (ray.slope - self.slope)

        if self.x0 <= col_x <= self.x1:
            return [(col_x, self.y_coords(col_x))]

        return None
