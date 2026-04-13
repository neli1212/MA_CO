import kagglehub
import shutil
import os

path = kagglehub.dataset_download("ambityga/imagenet100")
target = "../data/imagenet100/versions/8"
os.makedirs(os.path.dirname(target), exist_ok=True)
shutil.copytree(path, target, dirs_exist_ok=True)
print(f"Daten erfolgreich nach {target} kopiert.")