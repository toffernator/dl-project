from pathlib import Path
from src.preprocessing.minMaxScaling import MinMaxScalerBatched
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


if __name__ == "__main__":
    main()
