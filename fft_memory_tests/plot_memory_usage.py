import json
import matplotlib.pyplot as plt
import numpy as np
import glob

# Function to calculate theoretical memory usage
def theoretical_memory_1d(n1):
    # return (8 * n1 * 3 * 8) / (1024 * 1024)  # in MiB
    # return (2 * n1 * 8 + 2 * (n1 // 2 + 1) * 16 + n1 * 8) / (1024 * 1024)  # in MiB
    # Refined model
    initial_signals = 2 * n1
    fft_buffers = n1 + (n1 // 2 + 1)
    total_memory = initial_signals + fft_buffers
    return total_memory * 8 / (1024 ** 2)  # Convert to MB

def theoretical_memory_2d(n1, n2):
    # return (14 * n1 * n2 * 8) / (1024 * 1024)  # in MiB
    # return (2 * n1 * n2 * 8 + 2 * n1 * (n2 // 2 + 1) * 16 + n1 * n2 * 8) / (1024 * 1024)  # in MiB
    # Refined model
    initial_signals = 2 * n1 * n2
    fft_buffers = 2 * (n1 * (n2 // 2 + 1))
    total_memory = initial_signals + fft_buffers
    return total_memory * 8 / (1024 ** 2)  # Convert to MB

def theoretical_memory_3d(n1, n2, n3):
    # return (26 * n1 * n2 * n3 * 8) / (1024 * 1024)  # in MiB
    # return (2 * n1 * n2 * n3 * 8 + 2 * n1 * n2 * (n3 // 2 + 1) * 16 + n1 * n2 * n3 * 8) / (1024 * 1024)  # in MiB
    # Refined model
    initial_signals = 2 * n1 * n2 * n3
    fft_buffers = 2 * (n1 * n2 * (n3 // 2 + 1))
    total_memory = initial_signals + fft_buffers
    return total_memory * 8 / (1024 ** 2)  # Convert to MB

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
plt.show()