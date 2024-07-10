import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin
import argparse

# Define the original signal
def original_signal(t):
    return np.sin(t) + 0.5 * np.sin(3*t)

# Define parameters
N = 2  # Downsampling factor
fs = 100 # Sampling frequency
T = 2 * np.pi  # Period
t = np.linspace(-np.pi, np.pi, fs, endpoint=False)
x = (T/fs) * original_signal(t)

# Define a more complex filter (low-pass filter limited to [-π/N, π/N])
cutoff_freq = np.pi / N
numtaps = len(t)  # Length of the filter
nyquist_rate = fs / 2  # Nyquist rate is half the sampling frequency
filter_time = firwin(numtaps, cutoff=cutoff_freq/nyquist_rate, window="hamming")

# Fourier Transform function
def fourier_transform(signal):
    return np.fft.fftshift(np.fft.fft(signal))

# Low-pass filter
def low_pass_filter(signal, cutoff):
    spectrum = fourier_transform(signal)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), d=T/len(signal)))
    filtered_spectrum = spectrum * (np.abs(freqs) <= cutoff)
    return np.fft.ifft(np.fft.ifftshift(filtered_spectrum))

# Downsample function
def downsample(signal, factor):
    return signal[::factor]

# Upsample function
def upsample(signal, factor):
    upsampled = np.zeros(len(signal) * factor)
    upsampled[::factor] = signal
    return upsampled

# Plotting function for signal space
def plot_signal(ax, signal, label, color='blue'):
    ax.plot(np.linspace(-np.pi, np.pi, len(signal)), signal, linewidth=2, label=label, color=color)
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.grid(True)
    ax.legend()

# Plotting function for Fourier space
def plot_spectrum(ax, signal, label, color='blue'):
    spectrum = np.abs(fourier_transform(signal))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), d=T/len(signal)))
    ax.plot(freqs, spectrum, linewidth=2, label=label, color=color)
    ax.set_xlim([0, 8])
    ax.set_ylabel('Magnitude', fontsize=10)
    ax.grid(True)
    # ax.set_yscale("log")
    ax.legend()

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process some signals.")
parser.add_argument('-s', '--save', action='store_true', help='Save the figure')
parser.add_argument('-p', '--plot', action='store_true', help='Show the plot')
args = parser.parse_args()

# Create figure
fig, axs = plt.subplots(6, 2, figsize=(10, 15))

# First column titles
axs[0, 0].set_title('Time Domain', fontsize=12)
axs[0, 1].set_title('Frequency Domain', fontsize=12)

# Original signal and its Fourier transform
plot_signal(axs[0, 0], x, 'Original Signal')
plot_spectrum(axs[0, 1], x, 'Original Signal')
# Plot the filter in time and frequency domain
plot_signal(axs[0, 0], filter_time, 'Kernel h', color='C1')
plot_spectrum(axs[0, 1], filter_time, 'Kernel h', color='C1')

# Low-pass filtered signal
x_lp = low_pass_filter(x, np.pi/N)
plot_signal(axs[1, 0], x_lp, 'Low-pass Filtered Signal')
plot_spectrum(axs[1, 1], x_lp, 'Low-pass Filtered Signal')
plot_signal(axs[1, 0], filter_time, 'Kernel h', color='C1')
plot_spectrum(axs[1, 1], filter_time, 'Kernel h', color='C1')

# Downsampled signal
x_downsampled = downsample(x_lp, N)
plot_signal(axs[2, 0], x_downsampled, 'Downsampled Signal')
plot_spectrum(axs[2, 1], x_downsampled, 'Downsampled Signal')
filter_downsampled = downsample(filter_time, N)
plot_signal(axs[2, 0], filter_downsampled, 'Downsampled Kernel h', color='C1')
plot_spectrum(axs[2, 1], filter_downsampled, 'Downsampled Kernel h', color='C1')

# Convolved with kernel
x_convolved = np.convolve(x_downsampled, filter_downsampled, mode='same')
plot_signal(axs[3, 0], x_convolved, 'Convolved Signal', color='black')
plot_spectrum(axs[3, 1], x_convolved, 'Convolved Signal', color='black')

# Upsampled signal
x_upsampled = upsample(x_convolved, N)
plot_signal(axs[4, 0], x_upsampled, 'Upsampled Signal', color='black')
plot_spectrum(axs[4, 1], x_upsampled, 'Upsampled Signal', color='black')

# Low-pass filtered upsampled signal
x_lp_upsampled = low_pass_filter(x_upsampled, np.pi)
plot_signal(axs[5, 0], x_lp_upsampled, 'Low-pass Filtered Upsampled Signal', color='black')
plot_spectrum(axs[5, 1], x_lp_upsampled, 'Low-pass Filtered Upsampled Signal', color='black')

# Remove x-labels from all subplots except last row
for ax in axs[:-1, :].flatten():
    ax.set_xlabel('')

# Adjust layout for better readability
plt.tight_layout()
plt.subplots_adjust(top=0.95)
fig.suptitle('Effect of Different Operations on a 1D Signal', fontsize=16)

if args.save:
    plt.savefig('effect_of_operations_on_1D_signal.png', dpi=300)

if args.plot:
    plt.show()

# Optionally close the plot if running in a script
plt.close(fig)
