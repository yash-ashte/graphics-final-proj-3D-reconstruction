import numpy as np


wall_thickness = 0.15

cam_rad = 0.22


def dist_wall(px, pz, x1, z1, x2, z2, half_thickness):
    p1 = np.array([x1, z1])
    p2 = np.array([x2, z2])
    edge = p2 - p1
    length = np.linalg.norm(edge)
    if length < 1e-8:
        return float("inf")
    tangent = edge / length
    normal = np.array([-tangent[1], tangent[0]])
    rel = np.array([px, pz]) - p1
    u = float(np.dot(rel, tangent))
    v = float(np.dot(rel, normal))
    cu = np.clip(u, 0.0, length)
    cv = np.clip(v, -half_thickness, half_thickness)
    du, dv = u - cu, v - cv
    return du * du + dv * dv


def wall_hit(px, pz, walls, wall_thickness=wall_thickness, cam_rad=cam_rad):
    if not walls:
        return False
    half_t = wall_thickness * 0.5
    r2 = cam_rad * cam_rad
    for a, b in walls:
        d2 = dist_wall(px, pz, float(a[0]), float(a[1]), float(b[0]), float(b[1]), half_t)
        if d2 < r2:
            return True
    return False


def move(px, pz, dx, dz, walls, wall_thickness=wall_thickness, cam_rad=cam_rad):
    if not walls:
        return px + dx, pz + dz

   

    if not wall_hit(px + dx, pz + dz, walls):
        return px + dx, pz + dz
    
    if abs(dx) > 1e-8 and not wall_hit(px + dx, pz, walls):
        return px + dx, pz
    
    if abs(dz) > 1e-8 and not wall_hit(px, pz + dz, walls):
        return px, pz + dz
    
    return px, pz
