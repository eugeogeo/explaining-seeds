import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "./models/fine_tuned_resnet.pth"
test_images_dir = "./../testeImages"
input_size = 224
num_samples = 10

data_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ['01_intact', '02_cercospora', '03_greenish', '04_mechanical', '05_bug', '06_dirty', '07_humidity']
num_classes = len(class_names)

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

cam_extractor = GradCAM(model, target_layer="layer4")

def generate_gradcam_figure(image, image_path):
    input_tensor = data_transforms(image).unsqueeze(0).to(device)
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    pred_label = class_names[pred_class]

    activation_map = cam_extractor(pred_class, output)[0].cpu()
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() + 1e-8)

    original_image = image.resize((input_size, input_size))
    result = overlay_mask(original_image, transforms.ToPILImage()(activation_map), alpha=0.5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_title(f"Original - {os.path.basename(image_path)}")
    ax1.imshow(original_image)
    ax1.axis('off')

    ax2.set_title(f"GradCAM - Predição: {pred_label}")
    ax2.imshow(result)
    ax2.axis('off')

    return fig

image_paths = [os.path.join(test_images_dir, f)
               for f in os.listdir(test_images_dir)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:num_samples]

figures = []
for path in image_paths:
    image = Image.open(path).convert("RGB")
    figures.append(generate_gradcam_figure(image, path))

plt.show()
