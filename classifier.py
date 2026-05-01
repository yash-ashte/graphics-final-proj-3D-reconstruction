import torch
from torchvision import transforms, models # transform for image preprocessing, models for ResNet18
from PIL import Image #PIL for opening image files from disk

LABELS = ["office", "hallway", "apartment"] # builidng type outputs

def _load_model(model_path):
    """
    load trained cnn from .pth. build ResNet18, replace final layer to office, hallway, apartment, load saved weight
    set to evaluation so dropout/ batchworm behave correctly
    """
    model = models.resnet18(pretrained=False) # build ResNet18 architecture without weight
    model.fc = torch.nn.linear(model.fc.in_features, len(LABELS)) # replace final layer to output 3 classes
    model.load_state_dict(torch.load(model_path, map_location="cpu")) # load weights from .pth file
    model.eval() # set model t evaluation mode
    return model

def _preprocess_image(image_path):
    """
    prepare floorplan for cnn. resize to 244x244, convert to tensor, normalize pixel values, add batch dimension
    (1, 3, 244, 244) one image, 3 colors, 224x244 pixels
    """
    transform = transforms.Compose([
        transforms.Resize((224,224)), # resize image to 244 x 244
        transforms.ToTensor(), # convert PIL image to PyTorch tensor 
        transforms.Normalize ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # normalize each RGB channel to mean 0.5 and std 0.5)
    ])
    img = Image.open(image_path).convert("RGB") # open file and set to RGB color
    return transform(img).unsqueeze(0) # apply transforms and add batch demension (1, 3, 244, 244)

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

def _cnn_label(model_path, image_path):
    """
    call _load_model and _preprocess_image, run through image network, pick highest output score from 
    """
    model = _load_model(model_path) # load trained model
    tensor = _preprocess_image(image_path) # preprocess floor plan into a tensor
    with torch.no_grad(): #disable gradient tracking for inference
        output = model(tensor) # run image through tensor so output is (1,3)
        idx = output.argmax(dim=1).item() # convert index of highest scoring class to int
    return LABELS[idx] # return labe; string corresponding to predicted class

def classify_floorplan(wall_data, model_path=None, image_path=None):
    """
    Classifier API used by main.py.
    model_path is reserved for future CNN inference.
    """
    if model_path and image_path:
        # Placeholder: integrate TensorFlow/PyTorch model load + inference here.
        try:
            label = _cnn_label(model_path, image_path) # run cnn to get predictedt label
            return {
                "label": label, # builidng type string
                "material": _material_for_label(label), #material parameters for renderer
                "source": "cnn" # indicate cnn used
            }
        except Exception as e:
            print (f"[classifier] failed")

    label = _rule_based_label(wall_data)
    return {"label": label, "material": _material_for_label(label), "source": "rule_fallback"}
