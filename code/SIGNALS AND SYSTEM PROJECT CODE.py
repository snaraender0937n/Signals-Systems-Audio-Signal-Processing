import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal
from scipy.fft import fft, fftfreq
from fractions import Fraction
import os
import zipfile

# Utility function for resampling with explicit anti-aliasing filter
def resample_to(x, fs1, fs2, apply_antialiasing=True):
    """
    Resample signal from fs1 to fs2 with optional anti-aliasing filter.
    For educational purposes, we explicitly apply anti-aliasing filter.
    """
    if apply_antialiasing:
        # Apply anti-aliasing filter: cutoff at Nyquist frequency of target sampling rate
        nyquist_freq = fs2 / 2
        # Design anti-aliasing filter (low-pass filter at Nyquist)
        # Use a filter with sufficient attenuation
        numtaps = 101
        anti_alias_filter = signal.firwin(numtaps, nyquist_freq, fs=fs1, pass_zero='lowpass')
        # Apply filter before downsampling
        x_filtered = signal.lfilter(anti_alias_filter, 1, x)
    else:
        x_filtered = x
    
    # Perform resampling
    frac = Fraction(fs2, fs1).limit_denominator()
    return signal.resample_poly(x_filtered, up=frac.numerator, down=frac.denominator)

# Create output directory for plots
output_dir = "output_plots"
os.makedirs(output_dir, exist_ok=True)
plot_files = []  # Keep track of saved plot files

def plot_wave(sig, fs, title):
    t = np.arange(len(sig)) / fs
    plt.figure(figsize=(8,3))
    plt.plot(t, sig)
    plt.title(f"{title} - Time Domain")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{title.replace(' ', '_')}_time.png")
    plt.savefig(filename)
    plt.close()
    plot_files.append(filename)

def plot_fft(sig, fs, title):
    N = len(sig)
    yf = np.abs(fft(sig.astype(float)))
    xf = fftfreq(N, 1 / fs)
    plt.figure(figsize=(8,3))
    plt.plot(xf[:N//2], yf[:N//2])
    plt.title(f"{title} - Frequency Domain")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{title.replace(' ', '_')}_freq.png")
    plt.savefig(filename)
    plt.close()
    plot_files.append(filename)

def plot_fft_comparison(sig1, fs1, label1, sig2, fs2, label2, title):
    """Compare frequency spectra of two signals to visualize aliasing"""
    N1 = len(sig1)
    N2 = len(sig2)
    yf1 = np.abs(fft(sig1.astype(float)))
    yf2 = np.abs(fft(sig2.astype(float)))
    xf1 = fftfreq(N1, 1 / fs1)
    xf2 = fftfreq(N2, 1 / fs2)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(xf1[:N1//2], yf1[:N1//2], label=label1)
    plt.axvline(fs1/2, color='r', linestyle='--', label=f'Nyquist ({fs1/2} Hz)')
    plt.title(f"{label1} - Frequency Domain")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(xf2[:N2//2], yf2[:N2//2], label=label2, color='orange')
    plt.axvline(fs2/2, color='r', linestyle='--', label=f'Nyquist ({fs2/2} Hz)')
    plt.title(f"{label2} - Frequency Domain")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{title.replace(' ', '_')}_comparison.png")
    plt.savefig(filename)
    plt.close()
    plot_files.append(filename)

# Load audio file - try multiple possible paths
possible_paths = [
    r"E:/video_20250831_191054_edit.wav",
    "SIGNALS AND SYSTEM PROJECT ORIGINAL.wav",
    "original.wav"
]

signal_data = None
fs = None
filepath = None

for path in possible_paths:
    if os.path.exists(path):
        try:
            signal_data, fs = sf.read(path)
            filepath = path
            print(f"Loaded audio file: {path}")
            break
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

if signal_data is None:
    raise FileNotFoundError("Could not find audio file. Please ensure the file exists.")

print("Original sampling rate:", fs)

# If stereo, convert to mono
if signal_data.ndim > 1:
    signal_data = signal_data.mean(axis=1)

# Save original file
sf.write('SIGNALS AND SYSTEM PROJECT ORIGINAL.wav', signal_data, fs)

# Part A: Sampling and Reconstruction
# First, analyze the original signal's frequency content
N_orig = len(signal_data)
yf_orig = np.abs(fft(signal_data))
xf_orig = fftfreq(N_orig, 1 / fs)
# Find maximum frequency with significant energy (above -40dB threshold)
max_energy_idx = np.argmax(yf_orig[1:N_orig//2]) + 1
max_freq_orig = abs(xf_orig[max_energy_idx])
print(f"Maximum significant frequency in original signal: {max_freq_orig:.2f} Hz")
print(f"Nyquist frequency for 12kHz sampling: {12000/2} Hz")
print(f"Nyquist frequency for 5kHz sampling: {5000/2} Hz")

# Check if aliasing will occur
if max_freq_orig > 12000/2:
    print("WARNING: Signal has content above 6kHz - aliasing may occur even at 12kHz!")
if max_freq_orig > 5000/2:
    print("WARNING: Signal has content above 2.5kHz - aliasing WILL occur at 5kHz!")

# Downsample to 12kHz - no aliasing expected (if signal is bandlimited)
signal_above = resample_to(signal_data, fs, 12000, apply_antialiasing=True)
# Downsample to 5kHz - aliasing expected (if signal has content above 2.5kHz)
signal_below = resample_to(signal_data, fs, 5000, apply_antialiasing=True)

# Verify aliasing by comparing spectra
def check_aliasing(original, downsampled, fs_orig, fs_new, title):
    """Check if aliasing occurred by comparing frequency spectra"""
    N_orig = len(original)
    N_ds = len(downsampled)
    
    # Compute FFTs
    yf_orig = np.abs(fft(original))
    yf_ds = np.abs(fft(downsampled))
    
    xf_orig = fftfreq(N_orig, 1 / fs_orig)
    xf_ds = fftfreq(N_ds, 1 / fs_new)
    
    # Check for energy above Nyquist in downsampled signal
    nyquist_ds = fs_new / 2
    nyquist_idx = int(nyquist_ds * N_ds / fs_new)
    
    # Energy in original above Nyquist
    orig_above_nyquist = np.sum(yf_orig[np.abs(xf_orig) > nyquist_ds])
    # Energy in downsampled (should be zero above Nyquist, but aliasing causes folding)
    ds_above_nyquist = np.sum(yf_ds[nyquist_idx:N_ds//2])
    
    print(f"\n{title}:")
    print(f"  Energy above Nyquist ({nyquist_ds} Hz) in original: {orig_above_nyquist:.2e}")
    print(f"  Energy above Nyquist in downsampled: {ds_above_nyquist:.2e}")
    
    if ds_above_nyquist > orig_above_nyquist * 0.1:  # Significant energy suggests aliasing
        print(f"  ALIASING DETECTED: Energy folded back from above Nyquist frequency")
    else:
        print(f"  No significant aliasing detected")
    
    return orig_above_nyquist, ds_above_nyquist

# Check for aliasing
check_aliasing(signal_data, signal_above, fs, 12000, "12kHz Downsampling")
check_aliasing(signal_data, signal_below, fs, 5000, "5kHz Downsampling")

# Reconstruct back to original fs
rec_high = resample_to(signal_above, 12000, fs, apply_antialiasing=False)
rec_low = resample_to(signal_below, 5000, fs, apply_antialiasing=False)
# Save reconstructed files
sf.write('SIGNALS AND SYSTEM PROJECT RECONSTRUCTED NO ALIASING.wav', rec_high, fs)
sf.write('SIGNALS AND SYSTEM PROJECT RECONSTRUCTED ALIASING.wav', rec_low, fs)

# Plot original and reconstructed waveforms and spectra
plot_wave(signal_data, fs, "Original Signal")
plot_fft(signal_data, fs, "Original Signal")
plot_wave(rec_high, fs, "Reconstructed (No Aliasing)")
plot_fft(rec_high, fs, "Reconstructed (No Aliasing)")
plot_wave(rec_low, fs, "Reconstructed (Aliasing)")
plot_fft(rec_low, fs, "Reconstructed (Aliasing)")

# Add comparison plots to visualize aliasing
plot_fft_comparison(signal_data, fs, "Original", signal_above, 12000, 
                   "Downsampled 12kHz", "Aliasing Analysis: 12kHz Downsampling")
plot_fft_comparison(signal_data, fs, "Original", signal_below, 5000, 
                   "Downsampled 5kHz", "Aliasing Analysis: 5kHz Downsampling")

# Part B: Filter Design and Convolution
np.random.seed(0)
noisy_signal = signal_data + 0.02 * np.random.randn(*signal_data.shape)
sf.write('SIGNALS AND SYSTEM PROJECT WITH NOISE.wav', noisy_signal, fs)

# FIR Low-pass filter (cutoff 6kHz)
fir_coeff = signal.firwin(numtaps=101, cutoff=6000, fs=fs)
# IIR Butterworth low-pass filter (cutoff 6kHz)
b, a = signal.butter(4, 6000 / (fs / 2), 'low')

# Apply FIR filter (time domain convolution)
filtered_fir = np.convolve(noisy_signal, fir_coeff, mode='same')

# Apply FIR filter (frequency domain multiplication)
H_fir = fft(fir_coeff, n=len(noisy_signal))
S = fft(noisy_signal)
filtered_fir_fd = np.real(np.fft.ifft(H_fir * S))

# Apply IIR filter
filtered_iir = signal.lfilter(b, a, noisy_signal)

# Save filtered output
sf.write('SIGNALS AND SYSTEM PROJECT FILTERED OUTPUT.wav', filtered_fir, fs)

# Plot filtered results
plot_wave(filtered_fir, fs, "FIR Filtered (Time Domain)")
plot_fft(filtered_fir, fs, "FIR Filtered (Time Domain)")
plot_wave(filtered_fir_fd, fs, "FIR Filtered (Freq Domain)")
plot_fft(filtered_fir_fd, fs, "FIR Filtered (Freq Domain)")
plot_wave(filtered_iir, fs, "IIR Filtered (Time Domain)")
plot_fft(filtered_iir, fs, "IIR Filtered (Time Domain)")

# Part C: Frequency Analysis of speech signal
plot_fft(signal_data, fs, "Original Speech Spectrum")

# Identify fundamental frequency (improved method)
N = len(signal_data)
yf = np.abs(fft(signal_data))
xf = fftfreq(N, 1 / fs)

# Method 1: Find peak in lower frequency range (typical for speech: 80-400 Hz)
# Focus on frequencies below 1000 Hz for fundamental frequency
low_freq_range = np.abs(xf) < 1000
low_freq_indices = np.where(low_freq_range)[0]
if len(low_freq_indices) > 0:
    fund_index_low = low_freq_indices[np.argmax(yf[low_freq_indices])]
    fund_freq_low = abs(xf[fund_index_low])
else:
    fund_freq_low = None

# Method 2: Autocorrelation method (more robust for fundamental frequency)
# This finds the period of the signal
autocorr = np.correlate(signal_data, signal_data, mode='full')
autocorr = autocorr[len(autocorr)//2:]
# Find peaks in autocorrelation (excluding zero lag)
min_period = int(fs / 1000)  # Minimum period for 1000 Hz
max_period = int(fs / 80)    # Maximum period for 80 Hz
if max_period < len(autocorr):
    autocorr_range = autocorr[min_period:max_period]
    if len(autocorr_range) > 0:
        peak_idx = np.argmax(autocorr_range) + min_period
        fund_freq_autocorr = fs / peak_idx
    else:
        fund_freq_autocorr = None
else:
    fund_freq_autocorr = None

# Use the more reliable method
if fund_freq_autocorr is not None and 80 <= fund_freq_autocorr <= 400:
    fund_freq = fund_freq_autocorr
    print(f"Fundamental frequency (autocorrelation method): {fund_freq:.2f} Hz")
elif fund_freq_low is not None and 80 <= fund_freq_low <= 400:
    fund_freq = fund_freq_low
    print(f"Fundamental frequency (FFT peak method): {fund_freq:.2f} Hz")
else:
    # Fallback to original method
    fund_index = np.argmax(yf[1:N//2]) + 1
    fund_freq = abs(xf[fund_index])
    print(f"Fundamental frequency (fallback method): {fund_freq:.2f} Hz")
    print("Note: This might be a harmonic, not the fundamental frequency")

# Compare original vs filtered speech
plot_fft(filtered_fir, fs, "Filtered Speech Spectrum")
plot_wave(filtered_fir, fs, "Filtered Speech Waveform")

# Create ZIP archive of all plots
zip_filename = "output_graphs.zip"
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for file in plot_files:
        zipf.write(file)
print(f"All plots saved and zipped into {zip_filename}")