import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models, datasets
import matplotlib.pyplot as plt
from PIL import Image
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import warnings

warnings.filterwarnings("ignore")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "fine_tuned_inception.pth"
test_images_dir = "./../testeImages"
input_size = 299
num_samples = 10  # Número máximo de imagens

data_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ['01_intact', '02_cercospora', '03_greenish', '04_mechanical', '05_bug', '06_dirty', '07_humidity']
num_classes = len(class_names)


model = models.inception_v3(pretrained=False, aux_logits=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

    
cam_extractor = GradCAM(model, target_layer="Mixed_7c")

def generate_gradcam_figure(image, image_path, model, cam_extractor, class_names):
    input_tensor = data_transforms(image).unsqueeze(0).to(device)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    pred_label = class_names[pred_class]

    # Gera CAM
    activation_map = cam_extractor(pred_class, output)[0].cpu()
    
    # Normaliza o mapa de ativação
    activation_map = activation_map - activation_map.min()
    activation_map = activation_map / (activation_map.max() + 1e-8)

    # Imagem original redimensionada
    original_image = image.resize((input_size, input_size))

    # Overlay
    result = overlay_mask(original_image, transforms.ToPILImage()(activation_map), alpha=0.5)

    # Criar figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.set_title(f"Original - {os.path.basename(image_path)}")
    ax1.imshow(original_image)
    ax1.axis('off')

    ax2.set_title(f"GradCAM - Predição: {pred_label}")
    ax2.imshow(result)
    ax2.axis('off')

    return fig

image_paths = [
    os.path.join(test_images_dir, fname)
    for fname in os.listdir(test_images_dir)
    if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
]

image_paths = image_paths[:num_samples]


figures = []
for image_path in image_paths:
    original_image = Image.open(image_path).convert("RGB")
    fig = generate_gradcam_figure(original_image, image_path, model, cam_extractor, class_names)
    figures.append(fig)

plt.show()
