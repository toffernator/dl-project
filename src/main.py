from pathlib import Path
from src.preprocessing.minMaxScaling import MinMaxScalerBatched
from src.downsampling import Downsampler

import src.utils as utils
import src.dataset_loader as dsloader
import h5py


def main():
    train, test = dsloader.get_intra_dataset_files()

    # first train data in Intra: "dataset/Intra/train/rest_105923_1.h5"
    matrix = train[0].load()

    scaler = MinMaxScalerBatched()
    scaler.applyMinMaxScaling(matrix)
    print(matrix)
    reduced = Downsampler().chunk_and_average_columns(matrix, 2)    # Chunk size = 2 
    print(reduced)
    print(reduced.shape)


if __name__ == "__main__":
    main()
