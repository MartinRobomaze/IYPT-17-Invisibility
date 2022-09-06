import numpy as np


def critical_angle(refr_idx_1, refr_idx_2):
    """
    Computes the critical angle of total reflection.

    :param refr_idx_1: Refractive index of optical medium in which ray is present
    :param refr_idx_2: Refractive index of optical medium to which ray enters
    :return: Angle of total internal reflection
    """
    return np.arcsin(refr_idx_2 / refr_idx_1)


def snell_refraction(refr_idx_1, refr_idx_2, angle_1):
    """
    Computes the refraction angle of a ray passing through different optical mediums.

    :param refr_idx_1: Refractive index of optical medium in which ray is present
    :param refr_idx_2: Refractive index of optical medium to which ray enters
    :param angle_1: Angle of incidence of the ray
    :return: Angle of refraction of the ray
    """
    return np.arcsin(np.sin(angle_1) * refr_idx_1 / refr_idx_2)

