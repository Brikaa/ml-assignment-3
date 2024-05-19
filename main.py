import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

dataset_dir = "./dataset/"
dirs = [
    d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))
]
features = []
target_names = []

for dir_name in dirs:
    dir_path = os.path.join(dataset_dir, dir_name)

    for file_name in os.listdir(dir_path):
        print("Processing " + file_name)
        file_path = os.path.join(dir_path, file_name)
        image = Image.open(file_path)
        image = image.convert("L").resize((64, 64))
        image_array = np.asarray(image) / 255.0
        features.append(image_array)
        target_names.append(dir_name)

features = np.array(features)

label_encoder = LabelEncoder()
targets = label_encoder.fit_transform(target_names)
