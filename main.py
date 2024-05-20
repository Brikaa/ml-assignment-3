import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from threading import Thread
import matplotlib.pyplot as plt
from keras import models, layers, optimizers, Input, losses

dataset_dir = "./dataset/"
dirs = [
    d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))
]
features = []
target_names = []


def preprocess_dir(dir_name, dir_path):
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        try:
            image = cv2.resize(image, (64, 64))
            image_normalized = image / 255.0
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
    t = Thread(
        target=preprocess_dir,
        args=[dir_name, dir_path],
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
# features_train = features_train[:19200]
# targets_train = targets_train[:19200]
# features_test = features_test[:4800]
# targets_test = targets_test[:4800]

print("Training SVM model")


def train_svm():
    clf = svm.SVC()
    clf.fit(features_train, targets_train)

    predicted = clf.predict(features_test)
    print(f"Accuracy: {metrics.accuracy_score(targets_test, predicted)}")
    print(f"F1 scores:\n{metrics.f1_score(targets_test, predicted, average=None)}")
    print(
        "F1 'weighted' average score: "
        + str(metrics.f1_score(targets_test, predicted, average="weighted"))
    )
    print(f"Confusion matrix:\n{metrics.confusion_matrix(targets_test, predicted)}")


# train_svm()

print("Training a feed-forward neural network model with back propagation")
features_test, features_validation, targets_test, targets_validation = train_test_split(
    features_test, targets_test, test_size=0.5
)


def train_mlp(model, no_epochs):
    history = model.fit(
        features_train,
        targets_train,
        validation_data=(features_validation, targets_validation),
        epochs=no_epochs,
    )
    training_accuracies = history.history["accuracy"]
    validation_accuracies = history.history["val_accuracy"]
    epochs = [i + 1 for i in range(no_epochs)]
    _, test_acc = model.evaluate(features_test, targets_test, verbose=2)
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
    return test_acc


def train_nn():
    model1 = models.Sequential(
        [
            Input(shape=(len(features_train[0]),)),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(len(np.unique(targets_train)), activation="softmax"),
        ]
    )
    model1.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    test_acc1 = train_mlp(model1, 17)

    model2 = models.Sequential(
        [
            Input(shape=(len(features_train[0]),)),
            layers.Dense(128, activation="sigmoid"),
            layers.Dense(128, activation="relu"),
            layers.Dense(len(np.unique(targets_train)), activation="sigmoid"),
        ]
    )
    model2.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    test_acc2 = train_mlp(model2, 20)

    best_model = model1
    if test_acc1 > test_acc2:
        print("Saving model1")
    else:
        print("Saving model2")
        best_model = model2

    # To form a probability distribution at the last layer

    best_model.save("./model.keras")
    loaded_model = models.load_model("./model.keras")
    predicted_arrs = loaded_model.predict(features_test)
    predicted = [np.argmax(p) for p in predicted_arrs]
    print(
        f"Accuracy on test dataset: {metrics.accuracy_score(targets_test, predicted)}"
    )
    print(f"F1 scores:\n{metrics.f1_score(targets_test, predicted, average=None)}")
    print(
        "F1 'weighted' average score: "
        + str(metrics.f1_score(targets_test, predicted, average="weighted"))
    )
    print(f"Confusion matrix:\n{metrics.confusion_matrix(targets_test, predicted)}")


train_nn()
