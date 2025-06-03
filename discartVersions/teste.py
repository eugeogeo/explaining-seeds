import torch
import torchvision.models as models

# Carrega o modelo InceptionV3
model = models.inception_v3(pretrained=True)

# Cria uma lista com os nomes das camadas
layer_names = []
for name, module in model.named_modules():
    layer_names.append(name)

# Caminho do arquivo de saída
output_path = "inceptionv3_layers_log.txt"

# Salva os nomes das camadas no arquivo
with open(output_path, "w") as f:
    for name in layer_names:
        f.write(name + "\n")

print(f"Arquivo de log salvo em: {output_path}")
