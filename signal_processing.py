import cv2
import numpy as np
import time
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fftpack import next_fast_len


class Signal_processing():
    def __init__(self):
        self.a = 1
        self.window_types = {
            'hamming': np.hamming,
            'hanning': np.hanning,
            'blackman': np.blackman,
            'kaiser': lambda N: np.kaiser(N, beta=14),  # Kaiser with beta=14 for good sidelobe suppression
            'flattop': signal.windows.flattop,
            'none': lambda N: np.ones(N)  # No window
        }
        
    def extract_color(self, ROIs):
        '''
        extract average value of green color from ROIs
        '''
        
        g = []
        for ROI in ROIs:
            g.append(np.mean(ROI[:,:,1]))
        output_val = np.mean(g)
        return output_val
    
    def normalization(self, data_buffer):
        '''
        normalize the input data buffer
        '''
        
        normalized_data = data_buffer/np.linalg.norm(data_buffer)
        
        return normalized_data
    
    def signal_detrending(self, data_buffer):
        '''
        remove overall trending
        
        '''
        detrended_data = signal.detrend(data_buffer)
        
        return detrended_data
        
    def interpolation(self, data_buffer, times, window_type='hamming'):
        '''
        interpolation data buffer to make the signal become more periodic (avoid spectral leakage)
        
        Parameters:
        -----------
        data_buffer : array-like
            Input signal data
        times : array-like
            Timestamps for the data points
        window_type : str, optional
            Type of window function to use ('hamming', 'hanning', 'blackman', 'kaiser', 'flattop', 'none')
            
        Returns:
        --------
        array-like
            Interpolated and windowed data
        '''
        L = len(data_buffer)
        
        even_times = np.linspace(times[0], times[-1], L)
        
        interp = np.interp(even_times, times, data_buffer)
        
        # Apply the selected window function
        if window_type in self.window_types:
            window_func = self.window_types[window_type]
            interpolated_data = window_func(L) * interp
        else:
            # Default to Hamming if window_type is not recognized
            interpolated_data = np.hamming(L) * interp
            
        return interpolated_data
        
    def fft(self, data_buffer, fps, window_type='hamming', buffer_size=None, pad_to_power_of_2=True):
        '''
        Perform FFT on the data buffer with optimized parameters
        
        Parameters:
        -----------
        data_buffer : array-like
            Input signal data
        fps : float
            Frames per second (sampling rate)
        window_type : str, optional
            Type of window function to use
        buffer_size : int, optional
            Size to resample the buffer to before FFT. If None, uses original size
        pad_to_power_of_2 : bool, optional
            Whether to zero-pad the signal to a length that is a power of 2,
            which can speed up FFT computation
            
        Returns:
        --------
        tuple
            (fft_of_interest, freqs_of_interest)
        '''
        # Apply window function directly if not already applied
        if window_type != 'none' and window_type in self.window_types:
            data_buffer = self.window_types[window_type](len(data_buffer)) * data_buffer
        
        # Resample if buffer_size is specified
        if buffer_size is not None and buffer_size != len(data_buffer):
            # Resample data to the specified buffer size
            original_indices = np.arange(len(data_buffer))
            new_indices = np.linspace(0, len(data_buffer) - 1, buffer_size)
            data_buffer = np.interp(new_indices, original_indices, data_buffer)
        
        L = len(data_buffer)
        
        # Optionally pad to a power of 2 for more efficient FFT
        if pad_to_power_of_2:
            nfft = next_fast_len(L)
            if nfft != L:
                data_buffer = np.pad(data_buffer, (0, nfft - L), mode='constant')
                L = nfft
        
        freqs = float(fps) / L * np.arange(L // 2 + 1)
        freqs_in_minute = 60. * freqs
        
        raw_fft = np.fft.rfft(data_buffer*60)
        fft = np.abs(raw_fft)**2
        
        # Focus on frequencies in the heart rate range (50-180 BPM)
        interest_idx = np.where((freqs_in_minute > 50) & (freqs_in_minute < 180))[0]
        
        # Avoid indexing issues
        if len(interest_idx) > 0:
            interest_idx_sub = interest_idx
            freqs_of_interest = freqs_in_minute[interest_idx_sub]
            fft_of_interest = fft[interest_idx_sub]
        else:
            freqs_of_interest = np.array([])
            fft_of_interest = np.array([])

        return fft_of_interest, freqs_of_interest
    
    
    def compare_window_functions(self, data_buffer, fps, times=None):
        '''
        Compare different window functions and their effect on the FFT results
        
        Parameters:
        -----------
        data_buffer : array-like
            Input signal data
        fps : float
            Frames per second
        times : array-like, optional
            Timestamps for the data points. If None, evenly spaced times are used.
            
        Returns:
        --------
        dict
            Dictionary with FFT results for each window type
        '''
        if times is None:
            times = np.arange(len(data_buffer)) / fps
            
        results = {}
        
        plt.figure(figsize=(12, 8))
        
        for i, window_type in enumerate(self.window_types.keys()):
            # Apply interpolation with the current window
            if times is not None:
                processed_data = self.interpolation(data_buffer, times, window_type)
            else:
                # Apply window directly if no interpolation needed
                processed_data = self.window_types[window_type](len(data_buffer)) * data_buffer
                
            # Compute FFT
            fft_values, fft_freqs = self.fft(processed_data, fps, window_type)
            
            # Store results
            results[window_type] = {
                'fft': fft_values,
                'freqs': fft_freqs
            }
            
            # Plot
            plt.subplot(len(self.window_types), 1, i+1)
            plt.plot(fft_freqs, fft_values)
            plt.title(f'Window: {window_type}')
            plt.xlabel('Frequency (BPM)')
            plt.ylabel('Power')
            
            # Find the peak frequency (heart rate)
            if len(fft_values) > 0:
                peak_idx = np.argmax(fft_values)
                peak_freq = fft_freqs[peak_idx]
                plt.axvline(x=peak_freq, color='r', linestyle='--', 
                           label=f'Peak: {peak_freq:.1f} BPM')
                plt.legend()
        
        plt.tight_layout()
        plt.savefig('window_comparison.png')
        plt.close()
        
        return results
    
    def analyze_buffer_size_tradeoff(self, data_buffer, fps, sizes=None):
        '''
        Analyze the tradeoff between FFT buffer size, frequency resolution, and computation time
        
        Parameters:
        -----------
        data_buffer : array-like
            Input signal data
        fps : float
            Frames per second
        sizes : list, optional
            List of buffer sizes to try. If None, uses powers of 2.
            
        Returns:
        --------
        dict
            Dictionary with analysis results for each buffer size
        '''
        # TODO: fix the function to allow for different buffer sizes, this now can not use
        if sizes is None:
            # Try powers of 2 up to the original buffer size
            max_power = int(np.log2(len(data_buffer))) + 1
            sizes = [2**i for i in range(7, max_power + 1)]  # Start from 128
            
        results = {}
        times = []
        resolutions = []
        buffer_sizes = []
        
        for size in sizes:
            start_time = time.time()
            fft_values, fft_freqs = self.fft(data_buffer, fps, buffer_size=size)
            compute_time = time.time() - start_time
            
            # Calculate frequency resolution
            resolution = 60 * fps / size
            
            results[size] = {
                'fft': fft_values,
                'freqs': fft_freqs,
                'time': compute_time,
                'resolution': resolution
            }
            
            times.append(compute_time)
            resolutions.append(resolution)
            buffer_sizes.append(size)
        
        # Plot results
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(buffer_sizes, times, 'o-')
        plt.title('Computation Time vs Buffer Size')
        plt.xlabel('Buffer Size')
        plt.ylabel('Time (s)')
        plt.xscale('log', base=2)
        
        plt.subplot(2, 1, 2)
        plt.plot(buffer_sizes, resolutions, 'o-')
        plt.title('Frequency Resolution vs Buffer Size')
        plt.xlabel('Buffer Size')
        plt.ylabel('Resolution (BPM)')
        plt.xscale('log', base=2)
        
        plt.tight_layout()
        plt.savefig('buffer_size_analysis.png')
        plt.close()
        
        return results
    
    def butter_bandpass_filter(self, data_buffer, lowcut, highcut, fs, order=5):
        '''
        
        '''
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        
        filtered_data = signal.lfilter(b, a, data_buffer)
        
        return filtered_data










