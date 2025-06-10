import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. CENTRAL DE CONFIGURAÇÃO
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = './dataset'
VAL_FOLD = 'Fold3'
BATCH_SIZE = 16 # Pode ser maior para avaliação, pois não há backpropagation

# Dicionário com as informações de cada modelo a ser avaliado
MODELS_TO_EVALUATE = {
    "ResNet50_v2": {
        "path": "./models/best_32_resnet_model_v2.pth",
        "model_type": "resnet50",
        "input_size": 224
    },
    "SqueezeNet_v2": {
        "path": "./models/best_32_squeezenet_model_v2.pth",
        "model_type": "squeezenet",
        "input_size": 224
    },
    "InceptionV3_v2": {
        "path": "./models/best_16_inception_model_v2.pth",
        "model_type": "inception",
        "input_size": 299
    }
}

# Obter as classes do dataset
val_path = os.path.join(BASE_DIR, VAL_FOLD)
class_names = datasets.ImageFolder(val_path).classes
num_classes = len(class_names)

# =============================================================================
# 2. FUNÇÕES AUXILIARES
# =============================================================================

def load_model(model_type, model_path, num_classes):
    """Carrega a arquitetura do modelo, modifica a camada final e carrega os pesos salvos."""
    model = None
    if model_type == "resnet50":
        model = models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, num_classes)
        )
    elif model_type == "squeezenet":
        model = models.squeezenet1_0()
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1))
        model.classifier[0].p = 0.5
    elif model_type == "inception":
      model = models.inception_v3(aux_logits=True) 
      
      # Modifica a camada principal
      num_ftrs = model.fc.in_features
      model.fc = nn.Sequential(
          nn.Dropout(p=0.5),
          nn.Linear(num_ftrs, num_classes)
      )
      
      # Modifica a camada auxiliar
      num_ftrs_aux = model.AuxLogits.fc.in_features
      model.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes)
    else:
        raise ValueError("Tipo de modelo desconhecido")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    return model

def get_predictions_and_labels(model, dataloader):
    """Roda o modelo no dataloader e retorna todas as previsões, scores e rótulos."""
    model.eval()
    all_labels = []
    all_preds = []
    all_scores = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)

            outputs = model(inputs)
            
            # Para o classification_report (precisão, recall, f1)
            _, preds = torch.max(outputs, 1)
            
            # Para a curva ROC/AUC, precisamos das probabilidades (scores)
            scores = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_scores)


def plot_roc_auc(y_true, y_score, class_names, model_name):
    """Calcula e plota a curva ROC e a AUC para cada classe."""
    # Binariza os rótulos verdadeiros para o formato multi-classe
    y_true_binarized = label_binarize(y_true, classes=range(len(class_names)))
    
    plt.figure(figsize=(12, 10))
    colors = sns.color_palette("husl", len(class_names))

    for i, color in zip(range(len(class_names)), colors):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'ROC de {class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (False Positive Rate)', fontsize=14)
    plt.ylabel('Taxa de Verdadeiros Positivos (True Positive Rate)', fontsize=14)
    plt.title(f'Curva ROC Multi-Classe - Modelo: {model_name}', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# =============================================================================
# 3. BLOCO DE EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    for model_name, config in MODELS_TO_EVALUATE.items():
        print("="*60)
        print(f"AVALIANDO MODELO: {model_name}")
        print("="*60)

        # 1. Preparar os dados de validação com a transformação correta
        val_transform = transforms.Compose([
            transforms.Resize((config['input_size'], config['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_dataset = datasets.ImageFolder(os.path.join(BASE_DIR, VAL_FOLD), transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        # 2. Carregar o modelo treinado
        print(f"Carregando pesos de: {config['path']}")
        model = load_model(config['model_type'], config['path'], num_classes)

        # 3. Obter previsões, scores e rótulos
        print("Realizando inferência no conjunto de validação...")
        y_true, y_pred, y_score = get_predictions_and_labels(model, val_loader)

        # 4. Exibir relatório de classificação (Precisão, Revocação, F1-Score)
        print("\n--- Relatório de Classificação ---")
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        print(report)

        # 5. Plotar a curva ROC e calcular AUC
        print("\n--- Gerando Curva ROC e AUC ---")
        plot_roc_auc(y_true, y_score, class_names, model_name)
        
        print(f"\nFim da avaliação para o modelo {model_name}.\n\n")