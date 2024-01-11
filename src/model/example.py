import tensorflow as tf

keras = tf.keras

from keras import models, layers, losses

import numpy as np
from src.dataset_loader import Label, DatasetFile, get_intra_dataset_files
from pathlib import Path
import os
import gc


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

    return splitted


def example_model():
    model = models.Sequential(
        [
            layers.Flatten(input_shape=(248, 3563)),
            layers.Dense(20, activation="relu"),
            layers.Dense(4, activation="relu"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    trainer(model, 10, 8)


def trainer(model, epochs, batch_size):
    train, test = get_intra_dataset_files()

    train_per_subject = split_per_subject(train)
    test_per_subject = split_per_subject(test)

    for epoch in range(epochs):
        for subject, files in train_per_subject.items():
            print(f"Epoch #{epoch} subject {subject} {len(files)}")

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

        if epoch % 5 != 0 or epoch == 0:
            continue

        print(f"Evaluation")

        for subject, files in test_per_subject.items():
            x_test = []
            y_test = []

            for file in files:
                x_test.append(file.load())
                y_test.append(label_to_vector(file.label))

            x_test = np.array(x_test)
            y_test = np.array(y_test)

            print(x_test.shape, y_test.shape)

            result = model.evaluate(x_test, y_test, batch_size=16)
            print(result)

            del x_test
            del y_test
            gc.collect()


"""
other ideas:

convert to TFRecords as part of preprocessing:
https://stackoverflow.com/questions/48889482/feeding-npy-numpy-files-into-tensorflow-data-pipeline

https://stackoverflow.com/questions/46820500/how-to-handle-large-amouts-of-data-in-tensorflow/47040165#47040165


"""
