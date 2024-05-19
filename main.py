import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from threading import Thread
import shutil

dataset_dir = "./dataset/"
dirs = [
    d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))
]
features = []
target_names = []


def preprocess_dir(dir_name, dir_path, preprocessed_dir_path):
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        try:
            image = cv2.resize(image, (64, 64))
            image_normalized = cv2.normalize(
                image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            )
            cv2.imwrite(
                os.path.join(preprocessed_dir_path, file_name), image_normalized
            )
            features.append(image_normalized)
            target_names.append(dir_name)

        except:
            print(file_path + " is bad")


threads = []

print("Preprocessing")
for dir_name in dirs:
    dir_path = os.path.join(dataset_dir, dir_name)
    if dir_path.endswith("-preprocessed"):
        continue
    preprocessed_dir_path = os.path.join(dataset_dir, dir_name + "-preprocessed")
    if os.path.exists(preprocessed_dir_path):
        shutil.rmtree(preprocessed_dir_path)
    os.mkdir(preprocessed_dir_path)
    t = Thread(
        target=preprocess_dir,
        args=[dir_name, dir_path, preprocessed_dir_path],
    )
    threads.append(t)
    t.start()

for thread in threads:
    thread.join()

features = np.array(features)

label_encoder = LabelEncoder()
targets = label_encoder.fit_transform(target_names)
print(len(features))
print(len(targets))

features_train, features_test, targets_train, targets_test = train_test_split(
    features, targets, test_size=80
)

print("Training SVM model")
clf = svm.SVC()
clf.fit(features_train, targets_train)


print("F1 scores: " + metrics.f1_score(targets_test, clf.predict(features_test)))
