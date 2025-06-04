import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models, datasets
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PIL import Image
import warnings

# Ignorar warnings
warnings.filterwarnings("ignore")

# Parâmetros
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "./models/fine_tuned_inception.pth"
data_dir = "./../SOYPR/Fold2"  # Fold de validação usado para testar
input_size = 299
num_samples = 10  # Número de imagens para testar com LIME

# Transforms (mesmos usados no treinamento)
data_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# Dataset e classes
dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
class_names = dataset.classes

# Criar modelo Inception v3 com aux_logits=True
num_classes = len(class_names)

model = models.inception_v3(pretrained=False, aux_logits=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Função auxiliar para LIME
def batch_predict(images):
    model.eval()
    batch = torch.stack([data_transforms(Image.fromarray(img)) for img in images], dim=0)
    batch = batch.to(device)

    with torch.no_grad():
        outputs = model(batch)
        if isinstance(outputs, tuple):  # Se tiver AuxLogits
            outputs = outputs[0]
        probs = torch.nn.functional.softmax(outputs, dim=1)
    return probs.cpu().numpy()

# LIME
explainer = lime_image.LimeImageExplainer()

print('total images:', num_samples)
for i in range(num_samples):
    image_path, label = dataset.samples[i]
    original_image = Image.open(image_path).convert("RGB")
    np_image = np.array(original_image)

    explanation = explainer.explain_instance(
        np_image,
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    # Visualização
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original_image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"LIME - {class_names[explanation.top_labels[0]]}")
    plt.imshow(mark_boundaries(temp, mask))
    plt.axis("off")

    plt.tight_layout()
    plt.show()
