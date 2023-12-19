import os
import numpy as np

class Downsampler():
    def __init__(self, data):
        ...


    def applyDownSampling(data, downsampling_factor):
        new_column_size = int(data.shape[1]*downsampling_factor)
        downsampled_data = data[:, ::int(1/downsampling_factor)]
        return downsampled_data, new_column_size