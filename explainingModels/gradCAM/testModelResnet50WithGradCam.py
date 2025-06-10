import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F # <-- Importar o functional para o softmax
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image
# =============================================================================
# MUDANÇA 1: Usar um método mais avançado como o Grad-CAM++
# =============================================================================
from torchcam.methods import GradCAMpp 
from torchcam.utils import overlay_mask
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "./models/best_32_resnet_model_v2.pth"
test_dir = "./testImages"
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
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5, inplace=True),
    nn.Linear(num_ftrs, num_classes)
)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Usando GradCAMpp na camada convolucional final da ResNet
cam_extractor = GradCAMpp(model, target_layer="layer4")

# =============================================================================
# MUDANÇA 2: Função de geração de figura refatorada para mais clareza
# =============================================================================
def generate_gradcam_figure(image, image_path):
    input_tensor = data_transforms(image).unsqueeze(0).to(device)
    
    # Faz a predição
    output = model(input_tensor)
    
    # Calcula as probabilidades com Softmax para saber a confiança
    probabilities = F.softmax(output, dim=1)[0]
    
    # Pega a classe predita (índice) e sua probabilidade (confiança)
    pred_class_idx = probabilities.argmax().item()
    pred_confidence = probabilities[pred_class_idx].item()
    pred_label = class_names[pred_class_idx]

    # Gera o mapa de ativação para a classe predita
    # O [0] é para pegar o primeiro (e único) mapa da lista retornada
    activation_map = cam_extractor(pred_class_idx, output)[0].cpu()

    # Redimensiona a imagem original para o plot
    original_image = image.resize((input_size, input_size))
    
    # Sobrepõe o mapa de calor na imagem original
    result = overlay_mask(
        original_image, 
        transforms.ToPILImage()(activation_map), 
        alpha=0.6, # Alpha um pouco maior para dar mais destaque ao mapa
        colormap='viridis' # Uma paleta de cores melhor
    )

    # Cria a figura para o plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Arquivo: {os.path.basename(image_path)}', fontsize=16)

    ax1.set_title("Original")
    ax1.imshow(original_image)
    ax1.axis('off')

    # Título mais informativo, com a classe e a confiança
    ax2.set_title(f"Grad-CAM++: {pred_label} ({pred_confidence:.2%})")
    ax2.imshow(result)
    ax2.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Ajusta o layout para o supertítulo
    return fig

# ... (o laço principal para salvar as figuras permanece o mesmo) ...
output_dir = "gradcam_results_improved"
os.makedirs(output_dir, exist_ok=True)
print(f"Os resultados do Grad-CAM melhorado serão salvos na pasta: {output_dir}")

image_paths = [os.path.join(test_dir, f)
               for f in os.listdir(test_dir)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:num_samples]

for path in image_paths:
    print(f"Processando imagem: {path}")
    image = Image.open(path).convert("RGB")
    fig = generate_gradcam_figure(image, path)
    
    base_filename = os.path.basename(path)
    save_path = os.path.join(output_dir, f"gradcam++_{base_filename}")
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

print("\nProcessamento concluído!")