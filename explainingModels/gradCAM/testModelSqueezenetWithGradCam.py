import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # Importar para usar o softmax
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

# --- Configurações ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "./models/best_32_squeezenet_model_v2.pth"
test_images_dir = "./testImages"  # Mantido o seu caminho
output_dir = "gradcam_result_squeezenet" # Pasta para salvar os resultados
input_size = 224
num_samples = 10

# --- Transformações e Nomes de Classes ---
data_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
class_names = ['01_intact', '02_cercospora', '03_greenish', '04_mechanical', '05_bug', '06_dirty', '07_humidity']
num_classes = len(class_names)

# --- Carregar Modelo SqueezeNet ---
model = models.squeezenet1_0(pretrained=False)
# Adaptação correta para o classificador da SqueezeNet
model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
model.num_classes = num_classes # Adicionado por consistência

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# --- Configurar Extrator Grad-CAM++ ---
# A camada "features.12" é a correta, pois é o último módulo Fire antes do classificador.
cam_extractor = GradCAMpp(model, target_layer="features.12")

# =============================================================================
# MUDANÇA 2: Função de geração de figura refatorada para mais clareza
# (Idêntica à versão melhorada da ResNet)
# =============================================================================
def generate_gradcam_figure(image, image_path):
    input_tensor = data_transforms(image).unsqueeze(0).to(device)
    
    output = model(input_tensor)
    probabilities = F.softmax(output, dim=1)[0]
    
    pred_class_idx = probabilities.argmax().item()
    pred_confidence = probabilities[pred_class_idx].item()
    pred_label = class_names[pred_class_idx]

    activation_map = cam_extractor(pred_class_idx, output)[0].cpu()

    original_image = image.resize((input_size, input_size))
    
    result = overlay_mask(
        original_image, 
        transforms.ToPILImage()(activation_map), 
        alpha=0.6,
        colormap='viridis'
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Arquivo: {os.path.basename(image_path)}', fontsize=16)

    ax1.set_title("Original")
    ax1.imshow(original_image)
    ax1.axis('off')

    ax2.set_title(f"Grad-CAM++: {pred_label} ({pred_confidence:.2%})")
    ax2.imshow(result)
    ax2.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# =============================================================================
# MUDANÇA 3: Laço principal para processar e SALVAR as imagens
# =============================================================================
os.makedirs(output_dir, exist_ok=True)
print(f"Os resultados do Grad-CAM para SqueezeNet serão salvos em: {output_dir}")

image_paths = [os.path.join(test_images_dir, f)
               for f in os.listdir(test_images_dir)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:num_samples]

for path in image_paths:
    try:
        print(f"Processando imagem: {path}")
        image = Image.open(path).convert("RGB")
        fig = generate_gradcam_figure(image, path)
        
        base_filename = os.path.basename(path)
        save_path = os.path.join(output_dir, f"gradcam++_{base_filename}")
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    except FileNotFoundError:
        print(f"  -> AVISO: Imagem não encontrada em {path}. Pulando.")
    except Exception as e:
        print(f"  -> ERRO ao processar {path}: {e}")


print("\nProcessamento concluído!")