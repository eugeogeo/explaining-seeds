import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split # Not used in this version, but kept for context
import copy

base_dir = './../SOYPR'  # Altere se necessário
train_folds = ['Fold1', 'Fold2']
val_fold = 'Fold3'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transformações (SqueezeNet1_0 espera imagens 224x224)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.CenterCrop(224),    
        transforms.RandomHorizontalFlip(),
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

# carrega os dados
def load_folds(folds, transform):
    datasets_list = []
    for fold in folds:
        fold_path = os.path.join(base_dir, fold)
        datasets_list.append(datasets.ImageFolder(fold_path, transform=transform))
    return torch.utils.data.ConcatDataset(datasets_list)

# carrega dados
train_dataset = load_folds(train_folds, data_transforms['train'])
val_dataset = load_folds([val_fold], data_transforms['val'])

# loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# modelo
from torchvision.models import SqueezeNet1_0_Weights # Import SqueezeNet weights
weights = SqueezeNet1_0_Weights.DEFAULT
model_ft = models.squeezenet1_0(weights=weights) # Load SqueezeNet1_0

# ajustar a última camada
# a SqueezeNet tem uma camada classificadora no final
# a última camada é um conv2d dentro do classificador
# precisamos substituir o conv2d final do classificador
num_classes = len(train_dataset.datasets[0].classes)  # usa o primeiro dataset para pegar as classes

# o classificador do SqueezeNet é um módulo sequencial
# a última camada é um Conv2d, precisamos substituí-lo
# o Conv2d original tem 1000 canais de saída para o ImageNet
# nós o substituímos por um novo Conv2d com canais de saída num_classes.
model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
# também, mude o dropout para 0.5 conforme o artigo original do SqueezeNet
model_ft.classifier[0].p = 0.5
# a ativação final do SqueezeNet é um ReLU, seguido por AdaptiveAvgPool2d
# não precisamos mudar o AdaptiveAvgPool2d
# o modelo espera uma saída final de (batch_size, num_classes, 1, 1) que é então comprimida

model_ft = model_ft.to(device)

# otimizador e critério
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.parameters(), lr=0.001)

# treinamento
num_epochs = 5
best_model_wts = copy.deepcopy(model_ft.state_dict())
best_acc = 0.0

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
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
            inputs = inputs.to(device)
            labels = labels.to(device)

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

        # para salvar melhor o modelo
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model_ft.state_dict())

# salvar o modelo

model_path = 'fine_tuned_squeezenet.pth'

model_ft.load_state_dict(best_model_wts)
torch.save(model_ft.state_dict(), model_path)
print(f'Modelo salvo em: {model_path}')
