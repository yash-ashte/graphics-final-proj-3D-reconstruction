def _rule_based_label(wall_data):
    walls = wall_data["walls"]
    bounds = wall_data["bounds"]
    width = max(bounds[1] - bounds[0], 1e-6)
    depth = max(bounds[3] - bounds[2], 1e-6)
    area = width * depth
    density = len(walls) / area

    if density > 0.9:
        return "office"
    if width > depth * 1.4 or depth > width * 1.4:
        return "hallway"
    return "apartment"


def _material_for_label(label):
    material_table = {
        "office": {"color": (0.9, 0.95, 1.0), "spec_strength": 0.45, "shininess": 32.0},
        "hallway": {"color": (0.95, 0.9, 0.85), "spec_strength": 0.2, "shininess": 8.0},
        "apartment": {"color": (1.0, 0.95, 0.9), "spec_strength": 0.3, "shininess": 16.0},
    }
    return material_table.get(label, material_table["apartment"])


def classify_floorplan(wall_data, model_path=None):
    """
    Classifier API used by main.py.
    model_path is reserved for future CNN inference.
    """
    if model_path:
        # Placeholder: integrate TensorFlow/PyTorch model load + inference here.
        pass

    label = _rule_based_label(wall_data)
    return {"label": label, "material": _material_for_label(label), "source": "rule_fallback"}
