import os
import subprocess
import shutil

DATASET_DIR = "dataset"
SOYPR_REPO = "https://github.com/julianofoleiss/SOYPR"

def clone_soypr_dataset():
    if os.path.exists(DATASET_DIR):
        overwrite = input(f"[AVISO] O diretório '{DATASET_DIR}' já existe. Deseja sobrescrever? (s/n): ").strip().lower()
        if overwrite != 's':
            print("[INFO] Operação cancelada pelo usuário.")
            return
        else:
            print("[INFO] Removendo diretório existente...")
            shutil.rmtree(DATASET_DIR)
    
    print("[INFO] Clonando repositório SOYPR temporariamente...")
    subprocess.run(["git", "clone", "--depth", "1", SOYPR_REPO, DATASET_DIR], check=True)

    print(f"[SUCESSO] Dataset copiado para './{DATASET_DIR}'.")

def create_dir_models():
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"[SUCESSO] Diretório '{models_dir}' criado.")
    else:
        print(f"[AVISO] O diretório '{models_dir}' já existe. Nenhuma ação foi tomada.")

if __name__ == "__main__":
    try:
        clone_soypr_dataset()
        create_dir_models()
    except Exception as e:
        print(f"[ERRO] {e}")
