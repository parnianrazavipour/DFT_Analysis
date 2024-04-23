
import matplotlib.pyplot as plt
import numpy as np
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import ipywidgets
import IPython.display as ipd
from flask import current_app as app

import matplotlib
matplotlib.use('Agg')


def save_figure(fig, filename):
    """Saves a matplotlib figure in the static directory."""
    static_folder = app.config.get('STATIC_FOLDER', '/default/path/if/not/set')
    path = os.path.join(static_folder, filename)
    try:
        fig.savefig(path)
        plt.close(fig)  
        app.logger.info(f'File saved successfully at {path}')
        return filename 
    except Exception as e:
        app.logger.error(f'Failed to save file {filename} at {path}: {str(e)}')
        raise

def create_combined_signal(waves):
    max_length = 0
    combined_signal = None

    for w in waves:

        freq, sampling_rate, duration, weight = w['freq'], w['rate'], w['duration'] , w['weight']
        print("duration", duration, type(duration), "sampling_rate", sampling_rate, type(sampling_rate))
        signal = create_sine_wave(freq,sampling_rate, duration)
        if combined_signal is None:
            combined_signal = signal
            max_length = len(signal)
        else:
            if len(signal) > max_length:
                combined_signal = np.pad(combined_signal, (0, len(signal) - max_length))
                max_length = len(signal)
            signal = np.pad(signal, (0, max_length - len(signal)))
            combined_signal += (signal * weight)

    return combined_signal, int(sampling_rate)


    
def load_audio(filename):
    try:
        signal, sample_rate = librosa.load(filename, sr=None, mono=False)
        return signal, sample_rate
    except Exception as e:
        print(f"An error occurred while loading the audio file: {e}")
        return None, None
    


def plot_time_domain(signal, sample_rate, start_sec=None, end_sec=None):
    print("checking:", start_sec, end_sec)
    
    num_channels = signal.shape[0] if signal.ndim > 1 else 1

    if start_sec is not None and end_sec is not None:
        num_plots_per_channel = 2
    else:
        num_plots_per_channel = 1

    fig, axs = plt.subplots(num_channels * num_plots_per_channel, 1, figsize=(15, 5 * num_channels))

    if num_channels == 1 and num_plots_per_channel == 1:
        axs = [axs]  

    for channel in range(num_channels):
        channel_index = channel * num_plots_per_channel
        channel_signal = signal[channel] if signal.ndim > 1 else signal

        librosa.display.waveshow(channel_signal, sr=sample_rate, ax=axs[channel_index])
        axs[channel_index].set_title(f'Channel {channel + 1} Full Signal - Time Domain')
        axs[channel_index].set_xlabel('Time (s)')
        axs[channel_index].set_ylabel('Amplitude')

        if start_sec is not None and end_sec is not None:
            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)
            zoomed_signal = channel_signal[start_sample:end_sample]
            librosa.display.waveshow(zoomed_signal, sr=sample_rate, ax=axs[channel_index + 1])
            axs[channel_index + 1].set_title(f'Channel {channel + 1} Zoomed Signal - Time Domain ({start_sec} to {end_sec} s)')
            axs[channel_index + 1].set_xlabel('Time (s)')
            axs[channel_index + 1].set_ylabel('Amplitude')

    plt.tight_layout()
    return save_figure(fig, 'time_domain_plot.png')

def create_sine_wave(freq, sampling_rate, total_time_in_secs = None):
    if total_time_in_secs is None:
        total_time_in_secs = 1 / freq
    time = np.arange(total_time_in_secs * sampling_rate) / sampling_rate
    sine_wave = np.sin(2 * np.pi * freq * time)
    return sine_wave



def plot_signal_to_axes(axis, waveform, sample_rate, plot_title=None):
    axis.plot(waveform, alpha=0.8) 
    axis.set_xlabel("Samples")
    axis.set_ylabel("Amplitude")

    if plot_title:
        axis.set_title(plot_title, y=1.1)

    axis.grid(True)

    time_axis = axis.twiny()
    time_axis.set_xlim(axis.get_xlim())

    primary_ticks = axis.get_xticks()[1:-1]
    time_labels = primary_ticks / sample_rate
    displayed_samples = time_axis.get_xlim()[1] - time_axis.get_xlim()[0]
    displayed_time = displayed_samples / sample_rate

    if displayed_time < 1:
        time_axis.set_xlabel("Time (ms)")
        formatted_time_labels = [f"{x * 1000:.1f}" for x in time_labels]
    else:
        time_axis.set_xlabel("Time (secs)")
        formatted_time_labels = [f"{x:.2f}" for x in time_labels]

    time_axis.set_xticks(primary_ticks)
    time_axis.set_xticklabels(formatted_time_labels)


def brute_force_dft(signal, sample_rate):
    num_channels = signal.shape[0] if signal.ndim > 1 else 1
    figure, axes = plt.subplots(num_channels * 2, 1, figsize=(15, 7 * num_channels))

    for channel_index in range(num_channels):
        single_channel_signal = signal[channel_index, :] if num_channels > 1 else signal

        nyquist_frequency = sample_rate // 2
        frequency_range = range(1, nyquist_frequency)


        frequency_correlation_pairs = []

        for test_frequency in frequency_range:
            test_sine_wave = np.sin(2 * np.pi * test_frequency * np.arange(len(single_channel_signal)) / sample_rate)
            correlation_value = np.correlate(single_channel_signal, test_sine_wave, 'full')

            frequency_correlation_pairs.append((test_frequency, np.max(correlation_value)))

        frequencies, correlation_values = zip(*frequency_correlation_pairs)
        frequencies = np.array(frequencies)
        normalized_correlation_values = np.array(correlation_values) / (nyquist_frequency - 1) * 2

        plot_signal_to_axes(axes[channel_index * 2], single_channel_signal, sample_rate, f"Channel {channel_index + 1}")
        axes[channel_index * 2].set_title(f"Channel {channel_index + 1}", y=1.2)

        axes[channel_index * 2 + 1].plot(frequencies, normalized_correlation_values, 'b-')
        axes[channel_index * 2 + 1].set_title(f"Channel {channel_index + 1} - Brute force DFT using cross-correlation")
        axes[channel_index * 2 + 1].set_xlabel("Frequency (Hz)")
        axes[channel_index * 2 + 1].set_ylabel("Normalized Cross-correlation")

        peak_indices, _ = sp.find_peaks(normalized_correlation_values, height=0.08)
        peak_frequencies = frequencies[peak_indices]
        peak_amplitudes = normalized_correlation_values[peak_indices]
        axes[channel_index * 2 + 1].plot(peak_frequencies, peak_amplitudes, 'rx')

        for idx, freq in enumerate(peak_frequencies):
            axes[channel_index * 2 + 1].text(freq + 2, peak_amplitudes[idx], f"{freq} Hz", color="red")

    figure.tight_layout(pad=2)
    return figure, axes, frequencies, normalized_correlation_values




def plot_frequency_domain_fft(signal, sample_rate, start_sec=None, end_sec=None, fft_pad_to=None):
    channels = signal.shape[0] if signal.ndim > 1 else 1
    fig, axs = plt.subplots(channels * 3, 1, figsize=(15, 7 * channels))  

    for channel in range(channels):
        channel_signal = signal[channel, :] if channels > 1 else signal
        
        if start_sec is not None and end_sec is not None:
            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)
            signal_window = channel_signal[start_sample:end_sample]
        else:
            signal_window = channel_signal

        # Perform FFT
        fft_spectrum = np.fft.fft(signal_window)
        freq = np.fft.fftfreq(len(fft_spectrum), 1 / sample_rate)

        magnitude_spectrum = np.abs(fft_spectrum)
        scaled_magnitude_spectrum = 2 * magnitude_spectrum / len(signal_window)
        decibel_spectrum = 20 * np.log10(scaled_magnitude_spectrum + np.finfo(float).eps)

        # Plot amplitude spectrum
        amplitude_ax = axs[channel * 3]
        amplitude_ax.plot(freq[:len(freq)//2], scaled_magnitude_spectrum[:len(freq)//2])
        amplitude_ax.set_title(f'Channel {channel + 1} - Amplitude Spectrum (numpy.fft)')
        amplitude_ax.set_xlabel('Frequency (Hz)')
        amplitude_ax.set_ylabel('Magnitude')

        # Plot decibel spectrum
        decibel_ax = axs[channel * 3 + 1]
        decibel_ax.plot(freq[:len(freq)//2], decibel_spectrum[:len(freq)//2])
        decibel_ax.set_title(f'Channel {channel + 1} - Decibel Spectrum')
        decibel_ax.set_xlabel('Frequency (Hz)')
        decibel_ax.set_ylabel('Magnitude (dB)')

        fft_pad_to = len(signal_window) * 4

        magnitude_spectrum_ax = axs[channel * 3 + 2]
        spectrum, freqs, line = magnitude_spectrum_ax.magnitude_spectrum(signal_window, Fs=sample_rate, color='r', pad_to=fft_pad_to)
        line.remove()
        spectrum = np.array(spectrum) * 2
        magnitude_spectrum_ax.plot(freqs, spectrum, color = 'r')
        magnitude_spectrum_ax.set_ylabel("Frequency Amplitude")
        
        magnitude_spectrum_ax.set_title(f'Channel {channel + 1} - Amplitude Spectrum (matplotlib.pyplot.magnitude_spectrum)')
        magnitude_spectrum_ax.set_xlabel('Frequency (Hz)')
        magnitude_spectrum_ax.set_ylabel('Amplitude')

        fft_peaks_indices, fft_peaks_props = sp.find_peaks(spectrum, height = 0.08)
        fft_freq_peaks = freqs[fft_peaks_indices]
        fft_freq_peaks_amplitudes = spectrum[fft_peaks_indices]

        magnitude_spectrum_ax.plot(fft_freq_peaks, fft_freq_peaks_amplitudes, linestyle='None', marker="x", color="black", alpha=0.8)
        for i in range(len(fft_freq_peaks)):
            magnitude_spectrum_ax.text(fft_freq_peaks[i] + 2, fft_freq_peaks_amplitudes[i], f"{fft_freq_peaks[i]} Hz", color="black")

    plt.tight_layout()
    fig_filename = 'frequency_domain_plot.png'
    return save_figure(fig, fig_filename) 



def process_file(file_path, start_sec = None, end_sec = None):
    signal, sample_rate = load_audio(file_path)
    if signal is not None and sample_rate is not None:
        time_domain_image = plot_time_domain(signal, sample_rate, start_sec, end_sec)
        frequency_domain_image = plot_frequency_domain_fft(signal, sample_rate, start_sec, end_sec)
        return time_domain_image, frequency_domain_image
    return None


def process_brute_force_dft(signal, sample_rate , plot_path, start_sec = None, end_sec = None):
    fig,axes, correlation_freqs, correlation_results = brute_force_dft(signal, sample_rate)
    fig.savefig(plot_path)
    plt.close(fig) 