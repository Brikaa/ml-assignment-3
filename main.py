import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import svm, metrics
from threading import Thread
import matplotlib.pyplot as plt
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
features = np.array([image.flatten() for image in features])

label_encoder = LabelEncoder()
targets = label_encoder.fit_transform(target_names)
print(len(features))
print(len(targets))
print(label_encoder.classes_)

features_train, features_test, targets_train, targets_test = train_test_split(
    features, targets, test_size=0.2
)

# TODO: remove
features_train = features_train[:1600]
targets_train = targets_train[:1600]
features_test = features_test[:400]
targets_test = targets_test[:400]

print("Training SVM model")
def train_svm():
    clf = svm.SVC()
    clf.fit(features_train, targets_train)

    predicted = clf.predict(features_test)
    print(f"Accuracy: {metrics.accuracy_score(targets_test, predicted)}")
    print(f"F1 scores:\n{metrics.f1_score(targets_test, predicted, average=None)}")
    print(f"F1 'weighted' average score: {metrics.f1_score(targets_test, predicted, average="weighted")}")
    print(f"Confusion matrix:\n{metrics.confusion_matrix(targets_test, predicted)}")

train_svm()

print("Training a feed-forward neural network model with back propagation")
features_test, features_validation, targets_test, targets_validation = train_test_split(features_test, targets_test, test_size=0.5)
def train_mlp():
    no_epochs = 100
    clf = MLPClassifier()
    training_accuracies = []
    validation_accuracies = []
    epochs = []
    classes = np.unique(targets_train)
    for epoch in range(no_epochs):
        clf.partial_fit(features_train, targets_train, classes=classes)
        training_predictions = clf.predict(features_train)
        validation_predictions = clf.predict(features_validation)
        training_accuracies.append(metrics.accuracy_score(targets_train, training_predictions))
        validation_accuracies.append(metrics.accuracy_score(targets_validation, validation_predictions))
        epochs.append(epoch + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_accuracies, label="training")
    plt.plot(epochs, validation_accuracies, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (% of samples)")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [100 - i for i in training_accuracies], label="training")
    plt.plot(epochs, [100 - i for i in validation_accuracies], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error (% of samples)")
    plt.legend()
    plt.show()

train_mlp()
