import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau # Importar o agendador
import copy
import numpy as np # Usado para o min_val_loss inicial

# =============================================================================
# 1. CENTRAL DE CONFIGURAÇÃO
# =============================================================================
# Mova todos os parâmetros que você pode querer ajustar para o topo.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = './dataset'
TRAIN_FOLDS = ['Fold1', 'Fold2']
VAL_FOLD = 'Fold3'
MODEL_SAVE_PATH = './models/best_resnet_model.pth' # Salvar o melhor modelo

# Hiperparâmetros de Treinamento
NUM_EPOCHS = 200  # Máximo de épocas, mas o Early Stopping provavelmente irá parar antes.
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Parâmetros de Early Stopping
PATIENCE = 10  # Número de épocas a esperar sem melhora antes de parar.
# =============================================================================

# Transformações (ResNet espera imagens 224x224)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Adicionado Data Augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# Função para carregar os dados (sem alterações)
def load_folds(folds, transform):
    datasets_list = []
    for fold in folds:
        fold_path = os.path.join(BASE_DIR, fold)
        datasets_list.append(datasets.ImageFolder(fold_path, transform=transform))
    return torch.utils.data.ConcatDataset(datasets_list)

# Carrega os dados
train_dataset = load_folds(TRAIN_FOLDS, data_transforms['train'])
val_dataset = load_folds([VAL_FOLD], data_transforms['val'])

# Loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Carrega o modelo ResNet50 pré-treinado
weights = models.ResNet50_Weights.DEFAULT
model_ft = models.resnet50(weights=weights)

# Ajusta a última camada
num_ftrs = model_ft.fc.in_features
# Usa o primeiro dataset para pegar o número de classes
num_classes = len(train_dataset.datasets[0].classes)
model_ft.fc = nn.Linear(num_ftrs, num_classes)

model_ft = model_ft.to(DEVICE)

# Otimizador e critério
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)

# =============================================================================
# 2. LÓGICA DE EARLY STOPPING E LR SCHEDULER
# =============================================================================
# Agendador que reduz a taxa de aprendizado quando a 'val_loss' para de diminuir.
# scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Variáveis para o Early Stopping
min_val_loss = np.inf
early_stopping_counter = 0
best_model_wts = copy.deepcopy(model_ft.state_dict())
# =============================================================================


# Laço de treinamento
for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
    print('-' * 20)

    for phase in ['train', 'val']:
        if phase == 'train':
            model_ft.train()
            dataloader = train_loader
        else:
            model_ft.eval()
            dataloader = val_loader

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model_ft(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # =============================================================================
        # 3. VERIFICAÇÃO DO EARLY STOPPING E SALVAMENTO DO MODELO
        # =============================================================================
        if phase == 'val':
            # Atualiza o agendador de taxa de aprendizado com a perda de validação
            scheduler.step(epoch_loss)

            if epoch_loss < min_val_loss:
                # Se a perda de validação diminuiu, encontramos um modelo melhor.
                print(f'Validation loss decreased ({min_val_loss:.4f} --> {epoch_loss:.4f}). Saving model...')
                min_val_loss = epoch_loss
                early_stopping_counter = 0  # Reseta o contador
                best_model_wts = copy.deepcopy(model_ft.state_dict()) # Salva os pesos
            else:
                # Se a perda não diminuiu, incrementa o contador.
                early_stopping_counter += 1
                print(f'Early Stopping counter: {early_stopping_counter} out of {PATIENCE}')

    # Verifica se devemos parar o treinamento
    if early_stopping_counter >= PATIENCE:
        print('Early stopping triggered!')
        break # Sai do laço de épocas
    # =============================================================================


# Salva o melhor modelo encontrado durante o treinamento
print(f'Treinamento finalizado. Carregando o melhor modelo com val_loss: {min_val_loss:.4f}')
model_ft.load_state_dict(best_model_wts)
torch.save(model_ft.state_dict(), MODEL_SAVE_PATH)
print(f'Melhor modelo salvo em: {MODEL_SAVE_PATH}')