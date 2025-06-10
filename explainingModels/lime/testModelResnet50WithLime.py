import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model_path = "./models/fine_tuned_resnet.pth"
model_path = "./models/best_32_resnet_model_v2.pth"
test_dir = "./testImages"  # Pasta com imagens para testar
output_dir = "lime_results_resnet" # <-- 1. NOME DO DIRETÓRIO DE SAÍDA
input_size = 224

# --- Cria o diretório de saída se ele não existir ---
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Diretório '{output_dir}' criado.")

data_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ['01_intact', '02_cercospora', '03_greenish', '04_mechanical', '05_bug', '06_dirty', '07_humidity']
num_classes = len(class_names)

model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5, inplace=True),
    nn.Linear(num_ftrs, num_classes)
)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def batch_predict(images):
    model.eval()
    batch = torch.stack([data_transforms(Image.fromarray(img)) for img in images], dim=0).to(device)
    with torch.no_grad():
        outputs = model(batch)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    return probs.cpu().numpy()

explainer = lime_image.LimeImageExplainer()

image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Encontradas {len(image_files)} imagens para analisar.")

figures = []
image_filenames = []

for i in range(len(image_files)):
    image_path = image_files[i]
    image_filenames.append(os.path.basename(image_path)) # Salva o nome do arquivo original
    
    original_image = Image.open(image_path).convert("RGB")
    np_image = np.array(original_image)

    print(f"Processando imagem: {image_filenames[-1]}...")
    
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.set_title(f"Original - {os.path.basename(image_path)}")
    ax1.imshow(original_image)
    ax1.axis("off")

    ax2.set_title(f"LIME - {class_names[explanation.top_labels[0]]}")
    ax2.imshow(mark_boundaries(temp, mask))
    ax2.axis("off")

    figures.append(fig)

# --- Loop para salvar as figuras no diretório criado ---
print("Salvando as imagens de explicação do LIME...")
for idx, fig in enumerate(figures):
    # <-- 2. MONTA O CAMINHO COMPLETO (DIRETÓRIO + NOME DO ARQUIVO)
    # Usei o nome do arquivo original para facilitar a identificação
    base_filename = os.path.splitext(image_filenames[idx])[0]
    save_path = os.path.join(output_dir, f"lime_{base_filename}.png")
    
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig) # Fecha a figura para liberar memória

print(f"Resultados salvos em '{output_dir}'.")