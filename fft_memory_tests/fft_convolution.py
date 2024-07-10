import numpy as np
import scipy.fft as fft
import gc
import json
import sys

def fft_convolution_1d(n1):
    signal1 = np.random.rand(n1)
    signal2 = np.random.rand(n1)
    
    fft_signal1 = fft.rfftn(signal1)
    fft_signal2 = fft.rfftn(signal2)
    fft_signal1 *= fft_signal2
    ifft_result = fft.irfftn(fft_signal1, s=n1)
    
    return ifft_result

def fft_convolution_2d(n1, n2):
    signal1 = np.random.rand(n1, n2)
    signal2 = np.random.rand(n1, n2)
    
    fft_signal1 = fft.rfftn(signal1, axes=(0, 1))
    fft_signal2 = fft.rfftn(signal2, axes=(0, 1))
    fft_signal1 *= fft_signal2
    ifft_result = fft.irfftn(fft_signal1, s=(n1, n2), axes=(0, 1))
    
    return ifft_result

def fft_convolution_3d(n1, n2, n3):
    signal1 = np.random.rand(n1, n2, n3)
    signal2 = np.random.rand(n1, n2, n3)
    
    fft_signal1 = fft.rfftn(signal1, axes=(0, 1, 2))
    fft_signal2 = fft.rfftn(signal2, axes=(0, 1, 2))
    fft_signal1 *= fft_signal2
    ifft_result = fft.irfftn(fft_signal1, s=(n1, n2, n3), axes=(0, 1, 2))
    
    return ifft_result

def run_experiments(n):
    n_ = int(n ** (2/3))
    out1_1 = fft_convolution_1d(n_**3)
    out2_1 = fft_convolution_2d(n, n)
    out3_1 = fft_convolution_3d(n_, n_, n_)

if __name__ == "__main__":
    n = int(sys.argv[1])
    run_experiments(n)
