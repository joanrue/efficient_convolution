import json
import matplotlib.pyplot as plt
import numpy as np
import glob

# Function to calculate theoretical memory usage

def theoretical_peak_memory_1d(n1):
    padded_length = 2 * n1 - 1
    fft_length = math.ceil((2 * n1 - 1) / 2) + 1

    input_signal_size = 2 * padded_length
    fft_buffers_size = 2 * fft_length
    ifft_result_size = padded_length

    total_memory = input_signal_size + fft_buffers_size + ifft_result_size
    total_memory_mb = total_memory * 8 / 1024**2

    return total_memory_mb

def theoretical_peak_memory_2d(n1, n2):
    padded_size_1 = 2 * n1 - 1
    padded_size_2 = 2 * n2 - 1
    fft_size_2 = math.ceil((2 * n2 - 1) / 2) + 1

    input_matrix_size = 2 * padded_size_1 * padded_size_2
    fft_buffers_size = 2 * padded_size_1 * fft_size_2
    ifft_result_size = padded_size_1 * padded_size_2

    total_memory = input_matrix_size + fft_buffers_size + ifft_result_size
    total_memory_mb = total_memory * 8 / 1024**2

    return total_memory_mb

def theoretical_peak_memory_3d(n1, n2, n3):
    padded_size_1 = 2 * n1 - 1
    padded_size_2 = 2 * n2 - 1
    padded_size_3 = 2 * n3 - 1
    fft_size_3 = math.ceil((2 * n3 - 1) / 2) + 1

    input_tensor_size = 2 * padded_size_1 * padded_size_2 * padded_size_3
    fft_buffers_size = 2 * padded_size_1 * padded_size_2 * fft_size_3
    ifft_result_size = padded_size_1 * padded_size_2 * padded_size_3

    total_memory = input_tensor_size + fft_buffers_size + ifft_result_size
    total_memory_mb = total_memory * 8 / 1024**2

    return total_memory_mb

# Read JSON files
files = glob.glob("scalene/scalene_profile_n*.json")
measured_memory_1d = []
measured_memory_2d = []
measured_memory_3d = []
theoretical_memory_1d_values = []
theoretical_memory_2d_values = []
theoretical_memory_3d_values = []
n_values1d = []
n_values2d = []
n_values3d = []

for file in files:
    with open(file) as f:
        scalene_data = json.load(f)
        n = int(file.split('_n')[1].split('.')[0])
        n_ = int(n ** (2/3))
        functions = next(iter(scalene_data["files"].values()))["functions"]
        for func in functions:
            if func["line"] == "fft_convolution_1d":
                n_peak_mb = func["n_peak_mb"]
                
                n1 = n_ ** 3
                n_values1d.append(n1)
                measured_memory_1d.append(n_peak_mb)
                theoretical_memory_1d_values.append(theoretical_memory_1d(n1))
            elif func["line"] == "fft_convolution_2d":
                n_peak_mb = func["n_peak_mb"]
    
                n_values2d.append(n ** 2)
                measured_memory_2d.append(n_peak_mb)    
                theoretical_memory_2d_values.append(theoretical_memory_2d(n, n))
            elif func["line"] == "fft_convolution_3d":
                n_peak_mb = func["n_peak_mb"]
                n_values3d.append(n_ ** 3)                
                measured_memory_3d.append(n_peak_mb)
                theoretical_memory_3d_values.append(theoretical_memory_3d(n_, n_, n_))

ids = np.argsort(n_values1d)
# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(np.array(n_values1d)[ids], np.array(theoretical_memory_1d_values)[ids], c="C0", label='Theoretical 1D', marker='o')
plt.plot(np.array(n_values1d)[ids], np.array(measured_memory_1d)[ids], c="C0", ls="--", label='Measured 1D', marker='x')
plt.plot(np.array(n_values2d)[ids], np.array(theoretical_memory_2d_values)[ids], c="C1", label='Theoretical 2D', marker='o')
plt.plot(np.array(n_values2d)[ids], np.array(measured_memory_2d)[ids], c="C1", ls="--", label='Measured 2D', marker='x')
plt.plot(np.array(n_values3d)[ids], np.array(theoretical_memory_3d_values)[ids], c="C2", label='Theoretical 3D', marker='o')
plt.plot(np.array(n_values3d)[ids], np.array(measured_memory_3d)[ids], c="C2", ls="--", label='Measured 3D', marker='x')
plt.xlabel('n')
plt.ylabel('Memory Usage (MB)')
plt.xscale("log")
plt.yscale("log")
plt.title('Theoretical vs Measured Memory Usage in FFT Convolution')
plt.legend()


# Adjust layout for better readability
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig('memory_fft.png', dpi=300)
