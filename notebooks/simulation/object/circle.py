import numpy as np
import math
from .iface import Object
from ..utils.math import derivative
from ..ray import Ray


class Circle(Object):
    """
    Represents a circular part. Used to simulate circular lenses.
    """

    def __init__(self, x0, y0, radius, end_length, dir='U'):
        """
        Initializes a new circle.

        :param x0: X coordinate of circle origin
        :param y0: Y coordinate of circle origin
        :param radius: Circle radius
        :param end_length: Length of circular part
        """
        self.x0 = x0
        self.y0 = y0
        self.end_length = end_length
        self.r = radius
        self.dir = dir

    def y_coords(self, x):
        """
        Calculates the Y coordinate from a given X coordinate.

        :param x:
        :return:
        """
        if self.x0 - self.end_length / 2 <= x <= self.x0 + self.end_length / 2:
            if self.dir == 'U':
                return np.sqrt(self.r ** 2 - (x - self.x0) ** 2) + self.y0
            else:
                return -np.sqrt(self.r ** 2 - (x - self.x0) ** 2) + self.y0
        else:
            return None

    def normal_line_slope(self, x):
        """
        Calculates the slope of a normal line to the circle at point X.

        :param x: X coordinate of the normal line
        :return: Slope of the normal line
        """

        return -1 / derivative(self.y_coords, x) if derivative(self.y_coords, x) is not None else None

    def collides_with(self, ray: Ray):
        """
        Calculates the intersection points of a ray & a circle.

        :param ray:
        :return: Intersection points of ray and circle, None if there are none
        """
        if ray.slope is np.inf or ray.slope > 10 ** 9:
            y = self.y_coords(ray.x)
            if y is None:
                return None

            return [(ray.x, y)]

        ray_offset = ray.y - ray.slope * (ray.x - self.x0)

        # Calculation of the smallest distance between the ray and circle origin
        if ray_offset == 0:
            d = 0
        elif ray.slope == 0:
            d = ray_offset
        elif math.copysign(1, ray_offset) == math.copysign(1, ray.slope):
            d = -np.abs(
                ((-ray_offset / ray.slope) * ray_offset) / np.sqrt((-ray_offset / ray.slope) ** 2 + ray_offset ** 2))
        else:
            d = np.abs(
                ((-ray_offset / ray.slope) * ray_offset) / np.sqrt((-ray_offset / ray.slope) ** 2 + ray_offset ** 2))

        if np.abs(d) >= self.r:
            return None

        normal_line_angle = np.arctan(-1 / ray.slope)

        # Coordinates of point closest to circle origin.
        x_d = np.cos(normal_line_angle) * d
        y_d = np.sin(normal_line_angle) * d

        # Distance between point closest to circle origin and the intersection points.
        d_i = np.sqrt(self.r ** 2 - d ** 2)

        slope_angle = np.arctan(ray.slope)

        x_1 = np.cos(slope_angle) * d_i + x_d + self.x0
        x_2 = np.cos(slope_angle) * (-d_i) + x_d + self.x0

        y_1 = np.sin(slope_angle) * d_i + y_d + self.y0
        y_2 = np.sin(slope_angle) * (-d_i) + y_d + self.y0

        if np.isnan(x_1) or np.isnan(x_2) or np.isnan(y_1) or np.isnan(y_2):
            return None

        if np.isclose(x_1, x_2):
            return [(x_1, y_1)]
        else:
            return [(x_1, y_1), (x_2, y_2)]
