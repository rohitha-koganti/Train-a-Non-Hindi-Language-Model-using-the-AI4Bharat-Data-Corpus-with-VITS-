import librosa
import numpy as np

# Load your audio file
audio_file = '281474976883728_f1099_chunk_0.wav'  # Ensure this path is correct
y, sr = librosa.load(audio_file, sr=None)

# Extract MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Print the shape of the MFCCs
print("MFCCs shape:", mfccs.shape)
