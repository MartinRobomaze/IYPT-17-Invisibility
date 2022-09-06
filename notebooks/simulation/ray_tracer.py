import numpy as np
import math

from .utils.optics import snell_refraction, critical_angle
from .ray import Ray


def get_collisions(ray, objects):
    collisions = []

    for obj in objects:
        collisions_obj = obj.collides_with(ray)

        if collisions_obj is None:
            continue

        for collision in collisions_obj:
            if np.nan in collision:
                continue

            collision_x = collision[0]
            collision_y = collision[1]

            if ray.direction == 'R' and round(collision_y, 10) > round(ray.y, 10):
                collisions.append((collision_x, collision_y, obj))
            elif ray.direction == 'L' and round(collision_y, 10) < round(ray.y, 10):
                collisions.append((collision_x, collision_y, obj))

    if ray.direction == 'R':
        collisions.sort(key=lambda col: col[1])
    else:
        collisions.sort(key=lambda col: col[1], reverse=True)

    return collisions


def calc_incindence_angle(collision, ray):
    norm_l_slope = collision[2].normal_line_slope(collision[0])
    if norm_l_slope is None:
        return None, None

    norm_l_angle = math.atan(norm_l_slope)

    if norm_l_angle < 0:
        norm_l_angle += math.pi

    ray_angle = math.atan(ray.slope)
    if ray_angle < 0:
        ray_angle += math.pi

    return ray_angle - norm_l_angle, norm_l_angle


def check_collisions(collisions, ray, end):
    angle_1, norm_l_angle = calc_incindence_angle(collisions[0], ray)
    if angle_1 is None and norm_l_angle is None:
        collisions.pop(0)
        if len(collisions) > 0:
            angle_1, norm_l_angle = check_collisions(collisions, ray, end)
        else:
            return None, None

    return angle_1, norm_l_angle


def trace_ray(ray: Ray, objects, refr_idx, end):
    collisions = get_collisions(ray, objects)

    if len(collisions) == 0:
        if ray.direction == 'R':
            ray.update(ray.calc_x(end), end, ray.slope, 'R')
        elif ray.direction == 'L':
            ray.update(ray.calc_x(-end), -end, ray.slope, 'L')
        return ray

    angle_1, norm_l_angle = check_collisions(collisions, ray, end)
    if angle_1 is None and norm_l_angle is None:
        if ray.direction == 'R':
            ray.update(ray.calc_x(end), end, ray.slope, 'R')
        elif ray.direction == 'L':
            ray.update(ray.calc_x(-end), -end, ray.slope, 'L')
        return ray

    angle_2 = 0

    reflection = False

    if ray.collisions % 2 == 0:
        if round(abs(angle_1), 5) >= round(critical_angle(1, refr_idx), 5):
            if not ray.reflection:
                ray.reflection = True
                ray.reflection_origin = (collisions[0][0], collisions[0][1])
            reflection = True
            angle_2 = -angle_1
        else:
            ray.reflection = False
            angle_2 = snell_refraction(1, refr_idx, angle_1)
    if ray.collisions % 2 == 1:
        if round(abs(angle_1), 5) >= round(critical_angle(refr_idx, 1), 5):
            if not ray.reflection:
                ray.reflection = True
                ray.reflection_origin = (collisions[0][0], collisions[0][1])
            reflection = True
            angle_2 = -angle_1
        else:
            ray.reflection = False
            angle_2 = snell_refraction(refr_idx, 1, angle_1)

    if norm_l_angle >= 0:
        ray_slope = math.tan(norm_l_angle + angle_2)
    else:
        ray_slope = math.tan(norm_l_angle - angle_2)

    if reflection and (ray_slope < 0 or ray.prev_dir == 'L') and ray.reflection_origin[0] - collisions[0][2].x0 < 0:
        ray.update(collisions[0][0], collisions[0][1], ray_slope, 'L', new_collision=not reflection)
    elif reflection and (ray_slope > 0 or ray.prev_dir == 'L') and ray.reflection_origin[0] - collisions[0][2].x0 > 0:
        ray.update(collisions[0][0], collisions[0][1], ray_slope, 'L', new_collision=not reflection)
    else:
        ray.update(collisions[0][0], collisions[0][1], ray_slope, ray.prev_dir, new_collision=not reflection)

    return trace_ray(ray, objects, refr_idx, end)