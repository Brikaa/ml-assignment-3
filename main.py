import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from threading import Thread

dataset_dir = "./dataset/"
dirs = [
    d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))
]
features = []
target_names = []


def preprocess_dir(dir_path):
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        image = Image.open(file_path)
        image = image.convert("L").resize((64, 64))
        image_array = np.asarray(image) / 255.0
        features.append(image_array)
        target_names.append(dir_name)


threads = []

print("Preprocessing")
for dir_name in dirs:
    dir_path = os.path.join(dataset_dir, dir_name)
    t = Thread(target=preprocess_dir, args=[dir_path])
    threads.append(t)
    t.start()

for thread in threads:
    thread.join()

features = np.array(features)

label_encoder = LabelEncoder()
targets = label_encoder.fit_transform(target_names)
