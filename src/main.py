from pathlib import Path
from src.preprocessing.minMaxScaling import MinMaxScalerBatched
import src.utils as utils
import h5py

def main():
    filename = Path("dataset/Intra/train/rest_105923_1.h5")
    scaler = MinMaxScalerBatched()
    with h5py.File(filename, "r") as f:
        dataset_name = utils.get_dataset_name(filename)
        matrix = f.get(dataset_name)[()]
        scaler.applyMinMaxScaling(matrix)
        print(matrix)

if __name__ == "__main__":
    main()
