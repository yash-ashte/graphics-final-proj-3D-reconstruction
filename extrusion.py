import numpy as np

wall_height = 3.0
wall_thickness = 0.15

def add(vertices, indices, p0, p1, p2, p3, normal, uv):
    start = len(vertices) // 8
    uvs = [(0.0, 0.0), (1.0 * uv, 0.0), (1.0 * uv, 1.0), (0.0, 1.0)]
    corners = [p0, p1, p2, p3]
    for i, p in enumerate(corners):
        vertices.extend([p[0], p[1], p[2], normal[0], normal[1], normal[2], uvs[i][0], uvs[i][1]])
    indices.extend([start, start + 1, start + 2, start, start + 2, start + 3])


def build_rm(walls, bounds, wall_height=wall_height, wall_thickness=wall_thickness, include_floor=True):
    vertices = []
    indices = []

    for (x1, z1), (x2, z2) in walls:
        p1 = np.array([x1, 0.0, z1])
        p2 = np.array([x2, 0.0, z2])
        dir_vec = p2 - p1
        length = np.linalg.norm(dir_vec)
        if length < 1e-5:
            continue
        dir_vec /= length
        n = np.array([-dir_vec[2], 0.0, dir_vec[0]])
        off = n * (wall_thickness * 0.5)

        a = p1 + off
        b = p2 + off
        c = p2 - off
        d = p1 - off

        at = a + np.array([0.0, wall_height, 0.0])
        bt = b + np.array([0.0, wall_height, 0.0])
        ct = c + np.array([0.0, wall_height, 0.0])
        dt = d + np.array([0.0, wall_height, 0.0])

        add(vertices, indices, a, b, bt, at, n, 1.0)
        add(vertices, indices, c, d, dt, ct, -n, 1.0)

    if include_floor:
        min_x, max_x, min_z, max_z = bounds
        y = 0.0
        f0 = np.array([min_x, y, min_z])
        f1 = np.array([max_x, y, min_z])
        f2 = np.array([max_x, y, max_z])
        f3 = np.array([min_x, y, max_z])
        add(vertices, indices, f0, f1, f2, f3, np.array([0.0, 1.0, 0.0]), 4.0)

    return np.array(vertices), np.array(indices)
