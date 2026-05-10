import torch
from torchvision import transforms, models 
from PIL import Image

# LABELS = ["office", "hallway", "apartment"]

# def _load_model(model_path):
   
#     model = models.resnet18(pretrained=False) 
#     model.fc = torch.nn.linear(model.fc.in_features, len(LABELS)) 
#     model.load_state_dict(torch.load(model_path, map_location="cpu"))
#     model.eval() 
#     return model

# def _preprocess_image(image_path):
  
#     transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
#     img = Image.open(image_path).convert("RGB") 
#     return transform(img).unsqueeze(0)

def temp_label(wall_data):
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


def temp_mat(label):
    material_table = {
        "office": {"color": (0.9, 0.95, 1.0), "spec_strength": 0.45, "shininess": 32.0},
        "hallway": {"color": (0.95, 0.9, 0.85), "spec_strength": 0.2, "shininess": 8.0},
        "apartment": {"color": (1.0, 0.95, 0.9), "spec_strength": 0.3, "shininess": 16.0},
    }
    return material_table.get(label, material_table["apartment"])

# def _cnn_label(model_path, image_path):
#     model = _load_model(model_path)
#     tensor = _preprocess_image(image_path)
#     with torch.no_grad(): 
#         output = model(tensor)
#         idx = output.argmax(dim=1).item()
#     return LABELS[idx]

def classify_floorplan(wall_data, model_path=None, image_path=None):
    # if model_path and image_path:
    #     try:
    #         label = _cnn_label(model_path, image_path) 
    #         return {
    #             "label": label,
    #             "material": temp_mat(label),
    #             "source": "cnn" 
    #         }
    #     except Exception as e:
    #         print (f"[classifier] failed")

    label = temp_label(wall_data)
    return {"label": label, "material": temp_mat(label), "source": "rule_fallback"}
