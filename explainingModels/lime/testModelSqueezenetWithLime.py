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

# parâmetros
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "./models/fine_tuned_squeezenet.pth" # Alterado para o modelo SqueezeNet
test_dir = "./../testeImages"  # Pasta com imagens para testar
input_size = 224 # Alterado para o tamanho de entrada da SqueezeNet

# transforms
data_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ['01_intact', '02_cercospora', '03_greenish', '04_mechanical', '05_bug', '06_dirty', '07_humidity']
num_classes = len(class_names)


# modelo
# carrega o modelo SqueezeNet1_0 sem pesos pré-treinados inicialmente
model = models.squeezenet1_0(pretrained=False)

# ajustar a última camada para corresponder ao número de classes
model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))

model.classifier[0].p = 0.5


# carrega os pesos do modelo treinado
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# função de predição para LIME
def batch_predict(images):
    model.eval()
    batch = torch.stack([data_transforms(Image.fromarray(img)) for img in images], dim=0).to(device)
    with torch.no_grad():
        outputs = model(batch)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    return probs.cpu().numpy()

# inicializa o LIME
explainer = lime_image.LimeImageExplainer()

# lista com todas as imagens da pasta `testes`
image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# número de imagens a usar (ou todas)
num_samples = len(image_files)

# armazenar figuras
figures = []

# loop nas imagens
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

# mostrar os resultados
plt.show()
