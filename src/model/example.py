import tensorflow as tf

keras = tf.keras
from keras import models, layers, losses

import numpy as np
import gc
from sklearn.metrics import confusion_matrix
from src.dataset_loader import Label, DatasetFile, get_intra_dataset_files


def label_to_vector(label: Label):
    y = np.zeros(4)
    y[label.value] = 1.0
    return y


def split_per_label(dataset_files: list[DatasetFile]) -> dict[Label, list[DatasetFile]]:
    splitted = {}
    for label in Label:
        splitted[label] = []

    for file in dataset_files:
        splitted[file.label].append(file)

    return splitted


def split_per_subject(dataset_files: list[DatasetFile]) -> dict[str, list[DatasetFile]]:
    splitted = {}

    for file in dataset_files:
        if not file.subject in splitted:
            splitted[file.subject] = []

        splitted[file.subject].append(file)

    # for subject, files in splitted.items():
    #     splitted[subject] = round_robin_label(files)

    return splitted


def round_robin_label(dataset_files):
    splitted = split_per_label(dataset_files)
    return [item for items in zip(*splitted.values()) for item in items]


def example_model():
    model = models.Sequential(
        [
            layers.Conv2D(10, (5, 5), activation="relu", input_shape=(248, 3563, 1)),
            layers.MaxPool2D((2, 2)),
            layers.Conv2D(10, (5, 5), activation="relu"),
            layers.MaxPool2D((2, 2)),
            layers.Flatten(),
            layers.Dense(15, activation="relu"),
            layers.Dense(4, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.summary()

    trainer(model, 20, 4)


def trainer(model, epochs, batch_size):
    train, test = get_intra_dataset_files()

    train_per_subject = split_per_subject(train)
    test_per_subject = split_per_subject(test)

    def evaluate_model():
        print(f"Evaluation")

        for _, files in test_per_subject.items():
            x_test = []
            y_test = []

            for file in files:
                x_test.append(file.load())
                y_test.append(label_to_vector(file.label))

            x_test = np.array(x_test)
            y_test = np.array(y_test)

            print(x_test.shape, y_test.shape)

            result = model.evaluate(x_test, y_test)
            print(result)

            predictions = model.predict(x_test)

            print("Conf. Matrix")

            prediction_label = np.argmax(predictions, axis=1)
            actual_label = np.array([file.label.value for file in files])

            confmat = confusion_matrix(
                actual_label, prediction_label, labels=np.arange(len(Label))
            )
            print(confmat)

            del x_test
            del y_test
            gc.collect()

    for epoch in range(epochs):
        for subject, files in train_per_subject.items():
            print(f"Epoch #{epoch} subject {subject}")

            x_train = []
            y_train = []

            for file in files:
                x_train.append(file.load())
                y_train.append(label_to_vector(file.label))

            x_train = np.array(x_train)
            y_train = np.array(y_train)

            print(x_train.shape, y_train.shape)

            model.fit(x_train, y_train, batch_size=batch_size, epochs=1)

            del x_train
            del y_train
            gc.collect()

        ## evaluation

        if epoch % 5 == 0 and epoch > 0:
            evaluate_model()

    evaluate_model()


"""
other ideas:

convert to TFRecords as part of preprocessing:
https://stackoverflow.com/questions/48889482/feeding-npy-numpy-files-into-tensorflow-data-pipeline

https://stackoverflow.com/questions/46820500/how-to-handle-large-amouts-of-data-in-tensorflow/47040165#47040165


"""
