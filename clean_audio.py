import librosa
import numpy as np
import scipy.signal
import soundfile as sf  # For saving the cleaned audio

# Step 1: Load the audio file
audio_file = '281474976883728_f1099_chunk_0.wav'
y, sr = librosa.load(audio_file, sr=None)

# Step 2: Define a function for noise reduction using a bandpass filter
def noise_reduction(audio, sr):
    lowcut = 300.0  # Low cut frequency in Hz
    highcut = 3000.0  # High cut frequency in Hz
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(1, [low, high], btype='band')
    filtered_audio = scipy.signal.filtfilt(b, a, audio)
    return filtered_audio

# Step 3: Apply noise reduction
y_cleaned = noise_reduction(y, sr)

# Step 4: Save the cleaned audio to a new file
cleaned_audio_file = 'cleaned_281474976883728.wav'
sf.write(cleaned_audio_file, y_cleaned, sr)

print(f'Cleaned audio saved as: {cleaned_audio_file}')
