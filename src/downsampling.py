import os
import numpy as np
<<<<<<< HEAD
from scipy.signal import resample
=======
from scipy.signal import resample, decimate
>>>>>>> 5e7902a1b954b1fdde71ad8b770f5d9fa8cc4a53


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
<<<<<<< HEAD
    

    def sciResample(self ,data , new_sampling_rate) :
        original_sampling_rate = 2034 

        num_samples = int(data.shape[1] * new_sampling_rate / original_sampling_rate)

        downsampled_data = np.zeros((data.shape[0], num_samples))

        for i in range(data.shape[0]):
            downsampled_data[i, :] = resample(data[i, :], num_samples)

        return downsampled_data
    

=======

    def decimate(self, data, factor):
        return decimate(data, factor, axis=1)
>>>>>>> 5e7902a1b954b1fdde71ad8b770f5d9fa8cc4a53
