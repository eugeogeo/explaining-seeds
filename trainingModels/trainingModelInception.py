import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import numpy as np

# =============================================================================
# 1. CENTRAL DE CONFIGURAÇÃO
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = './dataset'
TRAIN_FOLDS = ['Fold1', 'Fold2']
VAL_FOLD = 'Fold3'
MODEL_SAVE_PATH = './models/best_64_inception_model_v2.pth' # <-- MODIFICAÇÃO: Novo nome para o modelo melhorado

# --- Hiperparâmetros de Treinamento ---
NUM_EPOCHS = 200 # Aumentado para permitir que o early stopping decida o melhor momento
# InceptionV3 é maior, um batch size menor ajuda a evitar problemas de memória.
BATCH_SIZE = 64
# <-- MODIFICAÇÃO: Reduzida a taxa de aprendizado para um ajuste mais fino.
LEARNING_RATE = 0.0001
# <-- MODIFICAÇÃO: Adicionado weight decay para regularização.
WEIGHT_DECAY = 1e-4

# --- Parâmetros de Early Stopping ---
# <-- MODIFICAÇÃO: Paciência ajustada para um treinamento mais longo.
PATIENCE = 15
# =============================================================================


# =============================================================================
# 2. DATA AUGMENTATION E TRANSFORMAÇÕES
# =============================================================================
# InceptionV3 espera imagens de 299x299.
# <-- MODIFICAÇÃO: Data Augmentation mais robusto para combater o overfitting.
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # Adiciona rotações aleatórias
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1), # Adiciona variações de cor
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}
# =============================================================================


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


# =============================================================================
# 3. MODIFICAÇÃO E CONFIGURAÇÃO DO MODELO
# =============================================================================
# Carrega o modelo InceptionV3 pré-treinado
weights = models.Inception_V3_Weights.DEFAULT
model_ft = models.inception_v3(weights=weights)

# Ajusta o número de classes
num_classes = len(train_dataset.datasets[0].classes)

# <-- MODIFICAÇÃO: Adiciona uma camada de Dropout antes do classificador final
# para regularização, espelhando a melhoria feita no ResNet.
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Sequential(
    nn.Dropout(p=0.5, inplace=True),
    nn.Linear(num_ftrs, num_classes)
)

# InceptionV3 também tem uma saída auxiliar (AuxLogits) que precisa ser ajustada.
# Esta saída também ajuda na regularização durante o treinamento.
num_ftrs_aux = model_ft.AuxLogits.fc.in_features
model_ft.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes)

model_ft = model_ft.to(DEVICE)

# Otimizador e critério
criterion = nn.CrossEntropyLoss()
# <-- MODIFICAÇÃO: Otimizador Adam com a nova taxa de aprendizado e weight_decay.
optimizer = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# =============================================================================


# <-- MODIFICAÇÃO: Agendador de Taxa de Aprendizagem
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# <-- MODIFICAÇÃO: Variáveis para o Early Stopping
min_val_loss = np.inf
early_stopping_counter = 0
best_model_wts = copy.deepcopy(model_ft.state_dict())

# =============================================================================
# 4. LAÇO DE TREINAMENTO
# =============================================================================
for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
    print(f'Current learning rate: {optimizer.param_groups[0]["lr"]}')
    print('-' * 20)

    for phase in ['train', 'val']:
        if phase == 'train':
            model_ft.train() # Habilita o modo de treino (ativa AuxLogits, Dropout, etc.)
            dataloader = train_loader
        else:
            model_ft.eval() # Habilita o modo de avaliação (desativa AuxLogits, Dropout, etc.)
            dataloader = val_loader

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                # Lógica específica do InceptionV3:
                # No modo de treino, ele retorna duas saídas (principal e auxiliar).
                if phase == 'train':
                  outputs, aux_outputs = model_ft(inputs)
                  loss1 = criterion(outputs, labels)
                  loss2 = criterion(aux_outputs, labels)
                  # A loss final é uma soma ponderada das duas losses.
                  loss = loss1 + 0.4 * loss2
                else:
                    # No modo de avaliação, ele retorna apenas a saída principal.
                    outputs = model_ft(inputs)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        if phase == 'train':
             epoch_loss = running_loss / len(train_dataset)
             epoch_acc = running_corrects.double() / len(train_dataset)
        else:
             epoch_loss = running_loss / len(val_dataset)
             epoch_acc = running_corrects.double() / len(val_dataset)

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # <-- MODIFICAÇÃO: Lógica de Early Stopping e salvamento baseada na perda de validação
        if phase == 'val':
            scheduler.step(epoch_loss)

            if epoch_loss < min_val_loss:
                print(f'Validation loss decreased ({min_val_loss:.4f} --> {epoch_loss:.4f}). Saving model...')
                min_val_loss = epoch_loss
                early_stopping_counter = 0
                best_model_wts = copy.deepcopy(model_ft.state_dict())
            else:
                early_stopping_counter += 1
                print(f'Early Stopping counter: {early_stopping_counter} out of {PATIENCE}')

    # Verifica se devemos parar o treinamento
    if early_stopping_counter >= PATIENCE:
        print('Early stopping triggered!')
        break

# Carrega os pesos do melhor modelo e salva em disco
print(f'\nTraining finished. Loading best model with val_loss: {min_val_loss:.4f}')
model_ft.load_state_dict(best_model_wts)
torch.save(model_ft.state_dict(), MODEL_SAVE_PATH)
print(f'Best model saved to: {MODEL_SAVE_PATH}')