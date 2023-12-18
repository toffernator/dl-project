from pathlib import Path
import os
import numpy as np
import sklearn.preprocessing

import h5py


def get_dataset_name(file_name_with_dir: Path) -> str:
    file_name_without_dir = file_name_with_dir.name
    name_parts = file_name_without_dir.split("_")
    dataset_name = "_".join(name_parts[:-1])
    return dataset_name


def main():
    # filename = Path("dataset/Intra/train/rest_105923_1.h5")
    # with h5py.File(filename, "r") as f:
    #     dataset_name = get_dataset_name(filename)
    #     matrix = f.get(dataset_name)[()]

    #     print(type(matrix))
    #     print(matrix.shape)
    minList = []
    maxList = []
    filenames = os.scandir(Path("dataset/Intra/train/"))
    for filename in filenames:
        with h5py.File(filename, "r") as f:
            dataset_name = get_dataset_name(filename)
            matrix = f.get(dataset_name)[()]
            cMin = matrix.min(axis=1)
            cMax = matrix.max(axis=1)
            minList.append(cMin)
            maxList.append(cMax)
    minList = np.array(minList).min(axis=0)
    maxList = np.array(maxList).max(axis=0)

    filename = Path("dataset/Intra/train/rest_105923_1.h5")
    with h5py.File(filename, "r") as f:
        dataset_name = get_dataset_name(filename)
        matrix = f.get(dataset_name)[()]
        minMaxScaling(matrix, minList, maxList)
        print(matrix)

    # print(minList)
    # print(maxList)

def minMaxScaling(data, minList, maxList):
    for (i, column) in enumerate(data):
        # foo = (column - minList[i])
        # if(np.all(np.greater(column, np.arange(minList[i], len(column)-1)))):
            # print("wrong min value")
            # exit()
        scaled_column = (column - minList[i]) / (maxList[i] - minList[i])
        data[i] = scaled_column




if __name__ == "__main__":
    main()
