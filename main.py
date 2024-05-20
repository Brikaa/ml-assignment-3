import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from threading import Thread
import matplotlib.pyplot as plt
import keras

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

# 80-10-10 split
# We set aside the validation data from the beginning to avoid decreasing the test size between different models
features_train, features_test, targets_train, targets_test = train_test_split(
    features, targets, test_size=0.2
)
features_test, features_validation, targets_test, targets_validation = train_test_split(
    features_test, targets_test, test_size=0.5
)

# TODO: remove
# features_train = features_train[:800]
# targets_train = targets_train[:800]
# features_test = features_test[:200]
# targets_test = targets_test[:200]

print("Training SVM model")


def get_and_print_metrics(predictions):
    accuracy = metrics.accuracy_score(targets_test, predictions)
    f1_all = metrics.f1_score(targets_test, predictions, average=None)
    f1_average = metrics.f1_score(targets_test, predictions, average="weighted")
    confusion_matrix = metrics.confusion_matrix(targets_test, predictions)
    print(f"Accuracy: {accuracy}")
    print(f"F1 scores:\n{f1_all}")
    print(f"F1 weighted average score: {f1_average}")
    print(f"Confusion matrix:\n{confusion_matrix}")
    return accuracy, f1_all, f1_average, confusion_matrix


def train_svm():
    clf = svm.SVC()
    clf.fit(features_train, targets_train)

    predicted = clf.predict(features_test)
    return get_and_print_metrics(predicted)


_, __, svm_f1, ___ = train_svm()

print("Training a feed-forward neural network model with back propagation")


def train_mlp(model, no_epochs):
    history = model.fit(
        features_train,
        targets_train,
        validation_data=(features_validation, targets_validation),
        epochs=no_epochs,
    )
    training_accuracies = history.history["accuracy"]
    validation_accuracies = history.history["val_accuracy"]
    training_loss = history.history["loss"]
    validation_loss = history.history["val_loss"]
    epochs = [i + 1 for i in range(no_epochs)]
    _, test_acc = model.evaluate(features_test, targets_test, verbose=2)
    plt.subplot(2, 2, 1)
    plt.plot(epochs, training_accuracies, label="training")
    plt.plot(epochs, validation_accuracies, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (% of samples)")
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(epochs, [100 - i for i in training_accuracies], label="training")
    plt.plot(epochs, [100 - i for i in validation_accuracies], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error (% of samples)")
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(epochs, training_loss, label="training")
    plt.plot(epochs, validation_loss, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    return test_acc


def train_nn():
    model1 = keras.models.Sequential(
        [
            keras.Input(shape=(len(features_train[0]),)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(len(np.unique(targets_train)), activation="softmax"),
        ]
    )
    model1.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    test_acc1 = train_mlp(model1, 17)

    model2 = keras.models.Sequential(
        [
            keras.Input(shape=(len(features_train[0]),)),
            keras.layers.Dense(128, activation="sigmoid"),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(len(np.unique(targets_train)), activation="sigmoid"),
        ]
    )
    model2.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    test_acc2 = train_mlp(model2, 20)

    best_model = model1
    if test_acc1 > test_acc2:
        print("Saving model 1 since it is the best based on test accuracy")
    else:
        print("Saving model 2 since it is the best based on test accuracy")
        best_model = model2

    # To form a probability distribution at the last layer

    best_model.save("./model.keras")
    loaded_model = keras.models.load_model("./model.keras")
    predicted_arrs = loaded_model.predict(features_test)
    predicted = [np.argmax(p) for p in predicted_arrs]
    return get_and_print_metrics(predicted)


_, __, nn_f1, ___ = train_nn()

print(
    f'{["SVM", "NN"][np.argmax([svm_f1, nn_f1])]} is the best model based on weighted average f1 score'
)
