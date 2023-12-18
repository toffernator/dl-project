from pathlib import Path
import os
import numpy as np

import h5py

def get_dataset_name(file_name_with_dir: Path) -> str:
    file_name_without_dir = file_name_with_dir.name
    name_parts = file_name_without_dir.split("_")
    dataset_name = "_".join(name_parts[:-1])
    return dataset_name

class MinMaxScalarBatched():
    def __init__(self, trainDir=Path("dataset/Intra/train/")):
        self.loadMinMaxList(self, trainDir)
    

    def loadMinMaxList(self, trainDir):
        minList = []
        maxList = []
        for filename in trainDir:
            with h5py.File(filename, "r") as f:
                dataset_name = get_dataset_name(filename)
                matrix = f.get(dataset_name)[()]
                cMin = matrix.min(axis=1)
                cMax = matrix.max(axis=1)
                minList.append(cMin)
                maxList.append(cMax)
        minList = np.array(minList).min(axis=0)
        maxList = np.array(maxList).max(axis=0)
        self.minList = minList
        self.maxList = maxList
    
    def applyMinMaxScaling(self, data):
        for (i, column) in enumerate(data):
            scaled_column = (column - self.minList[i]) / (self.maxList[i] - self.minList[i])
            data[i] = scaled_column
