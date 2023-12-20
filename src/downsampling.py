import os
import numpy as np
from scipy.signal import resample, decimate


class Downsampler:
    # def __init__(self, data):
    #     ...

    def applyDownSampling(data, downsampling_factor):
        new_column_size = int(data.shape[1] * downsampling_factor)
        downsampled_data = data[:, :: int(1 / downsampling_factor)]
        return downsampled_data, new_column_size

    def chunk_and_average_columns(self, data, chunk_size):
        num_chunks = data.shape[1] // chunk_size
        averaged_chunks = np.zeros((data.shape[0], num_chunks))

        for i in range(num_chunks):
            start_col = i * chunk_size
            end_col = start_col + chunk_size
            chunk = data[:, start_col:end_col]

            averaged_chunks[:, i] = np.mean(chunk, axis=1)

        return averaged_chunks

    def decimate(self, data, factor):
        return decimate(data, factor, axis=1)
