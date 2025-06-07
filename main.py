


#  interface simples para manipular o que o usuário quer fazer
# em construção

import os
import subprocess
import shutil
from trainingModels import trainingModelInception
from trainingModels import trainingModelResnet50
from trainingModels import trainingModelSqueezenet
 

def main():
    print("Bem-vindo ao sistema de treinamento e teste de modelos!")
    print("Você pode treinar ou testar um modelo de aprendizado de máquina.")
def get_user_choice():
    while True:
        choice = input("Você deseja treinar ou testar um modelo? (1 para treinar/2 para testar): ").strip().lower()
        if choice in ["1", "2"]:
            return "treinar" if choice == "1" else "testar"
        else:
            print("Opção inválida. Por favor, digite '1' para treinar ou '2' para testar.")
            
def get_model_choice():
    while True:
        model = input("Escolha o modelo (inception, resnet, squeezenet): ").strip().lower()
        if model in ["inception", "resnet", "squeezenet"]:
            return model
        else:
            print("Modelo inválido. Por favor, escolha 'inception', 'resnet' ou 'squeezenet'.")
            
def train_model(model):
    if model == "inception":
        trainingModelInception()
    elif model == "resnet":
        trainingModelResnet50()
    elif model == "squeezenet":
        trainingModelSqueezenet()
    else:
        print("Modelo desconhecido.")

# def get_test_choice()
        

def main():
    print("Bem-vindo ao sistema de treinamento e teste de modelos!")
    action = get_user_choice()
    
    if action == "treinar":
        model = get_model_choice()
        train_model(model)
    elif action == "testar":
        model = get_model_choice()
        test_model(model)
    print("Obrigado por usar o sistema!")

if __name__ == "__main__":
    main()