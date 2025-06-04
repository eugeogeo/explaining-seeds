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

# Ignorar warnings
warnings.filterwarnings("ignore")

# --- Parâmetros de Configuração ---
# Escolha o modelo a ser usado: 'inception' ou 'squeezenet'
MODEL_TO_USE = 'squeezenet' # Altere para 'inception' ou 'squeezenet'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_dir = "./../testeImages"  # Pasta com imagens para testar

class_names = ['01_intact', '02_cercospora', '03_greenish', '04_mechanical', '05_bug', '06_dirty', '07_humidity']
num_classes = len(class_names)

# --- Configurações Específicas do Modelo ---
if MODEL_TO_USE == 'inception':
    model_path = "./models/fine_tuned_inception.pth"
    input_size = 299
    # Modelo Inception v3
    model = models.inception_v3(pretrained=False, aux_logits=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
elif MODEL_TO_USE == 'squeezenet':
    model_path = "./models/fine_tuned_squeezenet.pth"
    input_size = 224
    # Modelo SqueezeNet1_0
    model = models.squeezenet1_0(pretrained=False)
    # Ajustar a última camada para corresponder ao número de classes
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model.classifier[0].p = 0.5 # Ajustar dropout se foi alterado durante o treinamento
else:
    raise ValueError("MODEL_TO_USE deve ser 'inception' ou 'squeezenet'")

# Transforms (adaptado ao input_size do modelo selecionado)
data_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Carregar os pesos do modelo treinado
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Função de predição para LIME
def batch_predict(images):
    model.eval()
    batch = torch.stack([data_transforms(Image.fromarray(img)) for img in images], dim=0).to(device)
    with torch.no_grad():
        outputs = model(batch)
        # Lidar com AuxLogits apenas se o modelo for Inception
        if MODEL_TO_USE == 'inception' and isinstance(outputs, tuple):
            outputs = outputs[0]
        probs = torch.nn.functional.softmax(outputs, dim=1)
    return probs.cpu().numpy()

# Inicializa o LIME
explainer = lime_image.LimeImageExplainer()

# Lista com todas as imagens da pasta `testes`
image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Número de imagens a usar (ou todas)
num_samples = len(image_files)

# Armazenar figuras
figures = []

# Loop nas imagens
print(f"Gerando explicações LIME para {num_samples} imagens usando {MODEL_TO_USE.upper()}...")
for i in range(num_samples):
    image_path = image_files[i]
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.set_title(f"Original - {os.path.basename(image_path)}")
    ax1.imshow(original_image)
    ax1.axis("off")

    ax2.set_title(f"LIME - {class_names[explanation.top_labels[0]]}")
    ax2.imshow(mark_boundaries(temp, mask))
    ax2.axis("off")

    figures.append(fig)

# Mostrar os resultados
plt.show()
