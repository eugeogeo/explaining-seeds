import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import copy

base_dir = './dataset'
train_folds = ['Fold1', 'Fold2']
val_fold = 'Fold3'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transformações (ResNet espera imagens 224x224)
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

# função para carregar os dados
def load_folds(folds, transform):
    datasets_list = []
    for fold in folds:
        fold_path = os.path.join(base_dir, fold)
        datasets_list.append(datasets.ImageFolder(fold_path, transform=transform))
    return torch.utils.data.ConcatDataset(datasets_list)

# carrega os dados
train_dataset = load_folds(train_folds, data_transforms['train'])
val_dataset = load_folds([val_fold], data_transforms['val'])

# loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# modelo
from torchvision.models import ResNet50_Weights # Importar os pesos da ResNet50
weights = ResNet50_Weights.DEFAULT
model_ft = models.resnet50(weights=weights) # Carregar ResNet50

# ajusta a última camada (camada 'fc' na ResNet)
num_ftrs = model_ft.fc.in_features
num_classes = len(train_dataset.datasets[0].classes)  # usa o primeiro dataset para pegar as classes
model_ft.fc = nn.Linear(num_ftrs, num_classes)

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

        # salva melhor modelo
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model_ft.state_dict())

# salvar o modelo

model_path = './models/fine_tuned_resnet.pth'

model_ft.load_state_dict(best_model_wts)
torch.save(model_ft.state_dict(), model_path)
print(f'Modelo salvo em: {model_path}')
