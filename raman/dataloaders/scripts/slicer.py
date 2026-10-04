
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def window_slicer(sample, window_size=None, stride=1, dilation=1):
    if window_size==None:
        window_size=sample.shape[-1]
    sliced_windows = sliding_window_view(sample, window_shape=window_size, axis=-1)
    sliced_windows = sliced_windows[:, ::stride, ::dilation]
    return np.array(sliced_windows, copy=True)