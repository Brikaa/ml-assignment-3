import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from threading import Thread, Lock
import matplotlib.pyplot as plt
import keras

dataset_dir = "./dataset/"
dirs = [
    d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))
]
grayscale_images = []
rgb_images = []
target_names = []
file_names = []

lock = Lock()


def preprocess_dir(dir_name, dir_path):
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        rgb_image = cv2.imread(file_path)
        try:
            grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            grayscale_image_resized = cv2.resize(grayscale_image, (64, 64))
            rgb_image_resized = cv2.resize(rgb_image, (64, 64))
            grayscale_image_normalized = grayscale_image_resized / 255.0
            rgb_image_normalized = rgb_image_resized / 255.0
            with lock:
                grayscale_images.append(grayscale_image_normalized)
                rgb_images.append(rgb_image_normalized)
                target_names.append(dir_name)
                file_names.append(file_name)

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

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(grayscale_images[i], cmap=plt.cm.binary)
    plt.xlabel(f"{file_names[i]}/{target_names[i]}")

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(rgb_images[i], cmap=plt.cm.binary)
    plt.xlabel(f"{file_names[i]}/{target_names[i]}")

gs_features = np.array(grayscale_images, dtype=np.float16)
gs_features = np.array([image.flatten() for image in gs_features], dtype=np.float16)
rgb_features = np.array(rgb_images, dtype=np.float16)
del grayscale_images
del rgb_images


label_encoder = LabelEncoder()
targets = label_encoder.fit_transform(target_names)
print(len(gs_features))
print(len(rgb_features))
print(len(targets))
print(label_encoder.classes_)

# 80-10-10 split
# We set aside the validation data from the beginning to avoid decreasing the test size between different models
(
    rgb_features_train,
    rgb_features_test,
    gs_features_train,
    gs_features_test,
    targets_train,
    targets_test,
) = train_test_split(rgb_features, gs_features, targets, test_size=0.2, random_state=12)
del rgb_features
del gs_features
del targets

gs_features_test, gs_features_validation = (
    gs_features_test[: len(gs_features_test) // 2],
    gs_features_test[len(gs_features_test) // 2 :],
)
rgb_features_test, rgb_features_validation = (
    rgb_features_test[: len(rgb_features_test) // 2],
    rgb_features_test[len(rgb_features_test) // 2 :],
)
targets_test, targets_validation = (
    targets_test[: len(targets_test) // 2],
    targets_test[len(targets_test) // 2 :],
)

# TODO: remove
rgb_features_train = rgb_features_train[:800]
rgb_features_test = rgb_features_test[:200]
gs_features_train = gs_features_train[:800]
gs_features_test = gs_features_test[:200]
targets_train = targets_train[:800]
targets_test = targets_test[:200]


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
    clf.fit(gs_features_train, targets_train)

    predicted = clf.predict(gs_features_test)
    return get_and_print_metrics(predicted)


_, __, svm_f1, ___ = train_svm()

print("Training a feed-forward neural network model with back propagation")


def plot_accuracy_and_error(
    no_epochs,
    training_accuracies,
    training_loss,
    validation_accuracies,
    validation_loss,
):
    epochs = [i + 1 for i in range(no_epochs)]
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, [i * 100 for i in training_accuracies], label="training")
    plt.plot(epochs, [i * 100 for i in validation_accuracies], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (% of samples)")
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(epochs, [100 - i * 100 for i in training_accuracies], label="training")
    plt.plot(epochs, [100 - i * 100 for i in validation_accuracies], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error (% of samples)")
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(epochs, training_loss, label="training")
    plt.plot(epochs, validation_loss, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()


def train_mlp(model, no_epochs):
    history = model.fit(
        gs_features_train,
        targets_train,
        validation_data=(gs_features_validation, targets_validation),
        epochs=no_epochs,
    )
    plot_accuracy_and_error(
        no_epochs,
        history.history["accuracy"],
        history.history["loss"],
        history.history["val_accuracy"],
        history.history["val_loss"],
    )
    _, test_acc = model.evaluate(gs_features_test, targets_test, verbose=2)
    return test_acc


def train_convolutional_mlp_rgb(model, no_epochs):
    history = model.fit(
        rgb_features_train,
        targets_train,
        validation_data=(rgb_features_validation, targets_validation),
        epochs=no_epochs,
    )
    plot_accuracy_and_error(
        no_epochs,
        history.history["accuracy"],
        history.history["loss"],
        history.history["val_accuracy"],
        history.history["val_loss"],
    )
    _, test_acc = model.evaluate(rgb_features_test, targets_test, verbose=2)
    return test_acc


def train_nn():
    model1 = keras.models.Sequential(
        [
            keras.Input(shape=(len(gs_features_train[0]),)),
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
            keras.Input(shape=(len(gs_features_train[0]),)),
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

    best_model.save("./model.keras")
    loaded_model = keras.models.load_model("./model.keras")
    predicted_arrs = loaded_model.predict(gs_features_test)
    predicted = [np.argmax(p) for p in predicted_arrs]
    return get_and_print_metrics(predicted)


_, __, nn_f1, ___ = train_nn()


def train_cnn():
    # https://www.tensorflow.org/tutorials/images/cnn
    # Common procedures and variables you might be interested in:
    # gs_features for grayscale features
    # rgb_features for rgb_features
    # plot_accuracy_and_error() for making a plot on the accuracy and error of the fitted model (check the train_mlp function)
    # save the best cnn model as "cnn-model.keras" since this file is gitignored
    # TODO: change, provide this function with the predicted values from the best model
    # Check train_nn functions for an example of how to determine the best model and how to save the model
    global gs_features_train, gs_features_test, gs_features_validation
    gs_features_train = gs_features_train.reshape((len(gs_features_train), 64, 64))
    gs_features_test = gs_features_test.reshape((len(gs_features_test), 64, 64))
    gs_features_validation = gs_features_validation.reshape(
        (len(gs_features_validation), 64, 64)
    )

    model1 = keras.models.Sequential()
    model1.add(keras.Input(shape=(64, 64, 1)))
    model1.add(
        keras.layers.Conv2D(8, 8, strides=(1, 1), padding="valid", activation="relu")
    )
    model1.add(keras.layers.MaxPool2D((2, 2)))
    model1.add(keras.layers.Conv2D(8, 8, activation="relu"))
    model1.add(keras.layers.MaxPool2D((2, 2)))
    model1.add(keras.layers.Flatten())
    model1.add(keras.layers.Dense(64, activation="relu"))
    model1.add(keras.layers.Dense(len(np.unique(targets_train)), activation="softmax"))

    model1.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    test_acc1 = train_mlp(model1, 20)

    model2 = keras.models.Sequential()
    model2.add(keras.Input(shape=(64, 64, 3)))
    model2.add(
        keras.layers.Conv2D(8, 8, strides=(1, 1), padding="valid", activation="relu")
    )
    model2.add(keras.layers.MaxPool2D((2, 2)))
    model2.add(keras.layers.Conv2D(8, 8, activation="relu"))
    model2.add(keras.layers.MaxPool2D((2, 2)))
    model2.add(keras.layers.Flatten())
    model2.add(keras.layers.Dense(64, activation="relu"))
    model2.add(keras.layers.Dense(len(np.unique(targets_train)), activation="softmax"))

    model2.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    test_acc2 = train_convolutional_mlp_rgb(model2, 20)

    best_model = model1
    best_model_is_rgb = False
    if test_acc1 > test_acc2:
        print("Saving grayscale CNN model since it is the best based on test accuracy")
    else:
        print("Saving RGB CNN model since it is the best based on test accuracy")
        best_model_is_rgb = True
        best_model = model2

    best_model.save("./cnn-model.keras")
    loaded_model = keras.models.load_model("./cnn-model.keras")
    if best_model_is_rgb:
        predicted_arrs = loaded_model.predict(rgb_features_test)
    else:
        predicted_arrs = loaded_model.predict(gs_features_test)

    predicted = [np.argmax(p) for p in predicted_arrs]
    return get_and_print_metrics(predicted)


print("Training a CNN model")
_, __, cnn_f1, ___ = train_cnn()

print(
    f'{["SVM", "NN", "CNN"][np.argmax([svm_f1, nn_f1, cnn_f1])]} is the best model based on weighted average f1 score'
)

plt.show()
