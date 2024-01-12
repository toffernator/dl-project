import tensorflow as tf

keras = tf.keras
from keras import models

import numpy as np
import gc
from sklearn.metrics import confusion_matrix
from src.dataset_loader import Label, DatasetFile

SAVED_MODEL_DIR = "models"


def train_eval(
    model: models.Model,
    epochs: int,
    batch_size: int,
    train_dataset: list[DatasetFile],
    test_dataset: list[DatasetFile],
    eval_per_epoch=5,
    save_model=True,
):
    def evaluate_model():
        print("Eval with test dataset")

        eval_results = []
        conf_matrix = np.zeros((len(Label), len(Label)), np.int32)

        for idx in range(0, len(test_dataset), batch_size):
            files = test_dataset[idx : idx + batch_size]

            x_test = []
            y_test = []

            for file in files:
                x_test.append(file.load())
                y_test.append(file.label.value)

            x_test = np.array(x_test)
            y_test = tf.one_hot(y_test, len(Label))

            eval_result = model.evaluate(x_test, y_test)
            eval_results.append(eval_result)

            predictions = model.predict(x_test)

            prediction_label = np.argmax(predictions, axis=1)
            actual_label = np.array([file.label.value for file in files])

            conf_matrix += confusion_matrix(
                actual_label, prediction_label, labels=np.arange(len(Label))
            )

            del x_test
            del y_test
            gc.collect()

        mean_results = np.mean(eval_results, axis=0)
        print(f"Err: {mean_results[0]} Acc: {mean_results[1]}")
        print("Conf. Matrix")
        print(conf_matrix)

    print("Start training")
    model.summary()

    train_dataset = round_robin(train_dataset, split_per_subject)

    for epoch in range(epochs):
        for idx in range(0, len(train_dataset), batch_size):
            files = train_dataset[idx : idx + batch_size]

            print(f"Epoch #{epoch} part #{idx // batch_size}")

            x_train = []
            y_train = []

            for file in files:
                x_train.append(file.load())
                y_train.append(file.label.value)

            x_train = np.array(x_train)
            y_train = tf.one_hot(y_train, len(Label))

            model.fit(x_train, y_train, batch_size=batch_size, epochs=1)

            del x_train
            del y_train
            gc.collect()

        ## evaluation

        if eval_per_epoch > 0 and epoch % eval_per_epoch == 0 and epoch > 0:
            evaluate_model()

    evaluate_model()

    model.save(f"{SAVED_MODEL_DIR}/{model.name}")


def train_eval_per_subject(
    model: models.Model,
    epochs: int,
    batch_size: int,
    train_dataset: list[DatasetFile],
    test_dataset: list[DatasetFile],
    eval_per_epoch=5,
    save_model=True,
):
    train_per_subject = split_per_subject(train_dataset)
    test_per_subject = split_per_subject(test_dataset)

    def evaluate_model():
        print("Eval with test dataset")

        for subject, files in test_per_subject.items():
            print(f"Test subject {subject}")

            x_test = []
            y_test = []

            for file in files:
                x_test.append(file.load())
                y_test.append(file.label.value)

            x_test = np.array(x_test)
            y_test = tf.one_hot(y_test, len(Label))

            eval_result = model.evaluate(x_test, y_test)

            predictions = model.predict(x_test)

            prediction_label = np.argmax(predictions, axis=1)
            actual_label = np.array([file.label.value for file in files])

            confmat = confusion_matrix(
                actual_label, prediction_label, labels=np.arange(len(Label))
            )

            print(f"Err: {eval_result[0]} Acc: {eval_result[1]}")
            print("Conf. Matrix")
            print(confmat)

            del x_test
            del y_test
            gc.collect()

    print("Start training")
    model.summary()

    for epoch in range(epochs):
        for subject, files in train_per_subject.items():
            print(f"Epoch #{epoch} subject {subject}")

            x_train = []
            y_train = []

            for file in files:
                x_train.append(file.load())
                y_train.append(file.label.value)

            x_train = np.array(x_train)
            y_train = tf.one_hot(y_train, len(Label))

            model.fit(x_train, y_train, batch_size=batch_size, epochs=1)

            del x_train
            del y_train
            gc.collect()

        ## evaluation

        if eval_per_epoch > 0 and epoch % eval_per_epoch == 0 and epoch > 0:
            evaluate_model()

    evaluate_model()

    model.save(f"{SAVED_MODEL_DIR}/{model.name}")


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


def round_robin(list, split_func):
    splitted = split_func(list)
    return [item for items in zip(*splitted.values()) for item in items]
