from pathlib import Path
import os
import numpy as np
import src.utils as utils
import h5py

class MinMaxScalerBatched():
    def __init__(self, trainDir=Path("dataset/Intra/train/")):
        self.loadMinMaxList(trainDir)
    

    def loadMinMaxList(self, trainDir):
        minList = []
        maxList = []
        filenames = os.scandir(trainDir)
        for filename in filenames:
            with h5py.File(filename, "r") as f:
                dataset_name = utils.get_dataset_name(filename)
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
