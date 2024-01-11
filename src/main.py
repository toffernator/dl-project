from pathlib import Path
from src.preprocessing.minMaxScaling import MinMaxScalerBatched
from src.downsampling import Downsampler

import src.utils as utils
from src.dataset_loader import get_intra_dataset_files, DatasetFile
import h5py

from src.model.example import example_model

import gc


def run_preprocess(train, test):
    scaler = MinMaxScalerBatched()
    sampler = Downsampler()

    def preprocess(file):
        matrix = file.load()
        scaler.applyMinMaxScaling(matrix)
        matrix = sampler.decimate(matrix, 10)
        file.save_preprocessed(matrix)

    for file in train:
        preprocess(file)

    for file in test:
        preprocess(file)


def main():
    train, test = get_intra_dataset_files()

    if not train[0].preprocessed:
        print("run preprocessing...")
        run_preprocess(train, test)

    example_model()

    # # first train data in Intra: "dataset/Intra/train/rest_105923_1.h5"
    # matrix = train[0].load()

    # scaler.applyMinMaxScaling(matrix)
    # print(matrix)
    # # reduced = Downsampler().chunk_and_average_columns(matrix, 2)    # Chunk size = 2
    # # print(reduced)
    # # print(reduced.shape)

    # sciReduced = Downsampler().sciResample(matrix, 1017)
    # print(sciReduced)
    # print(sciReduced.shape)


if __name__ == "__main__":
    main()
