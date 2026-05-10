import cv2
import numpy as np


def default():
    return [
        ((-4.0, -4.0), (4.0, -4.0)),
        ((4.0, -4.0), (4.0, 4.0)),
        ((4.0, 4.0), (-4.0, 4.0)),
        ((-4.0, 4.0), (-4.0, -4.0)),
    ]


def ext_walls(image_path):
    if not image_path:
        walls = default()
        return {"walls": walls, "bounds": (-4.0, 4.0, -4.0, 4.0)}

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read floorplan image: {image_path}")

    h, w = img.shape
    _, binary = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3))
    clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    lines = cv2.HoughLinesP(clean, 1, np.pi / 180.0, threshold=60, minLineLength=40, maxLineGap=10)
    if lines is None:
        walls = default()
        return {"walls": walls, "bounds": (-4.0, 4.0, -4.0, 4.0)}

    sx = 10.0 / max(float(w), 1.0)
    sz = 10.0 / max(float(h), 1.0)
    walls = []
    
    for l in lines[:, 0]:
        x1, y1, x2, y2 = l
        wx1 = (x1 - w * 0.5) * sx
        wz1 = (h * 0.5 - y1) * sz
        wx2 = (x2 - w * 0.5) * sx
        wz2 = (h * 0.5 - y2) * sz
        walls.append(((float(wx1), float(wz1)), (float(wx2), float(wz2))))

    xs = [p[0] for seg in walls for p in seg]
    zs = [p[1] for seg in walls for p in seg]
    bounds = (min(xs), max(xs), min(zs), max(zs))
    return {"walls": walls, "bounds": bounds}
