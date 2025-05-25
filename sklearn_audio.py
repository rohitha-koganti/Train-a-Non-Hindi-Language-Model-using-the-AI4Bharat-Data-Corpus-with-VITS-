import librosa
import numpy as np
from sklearn.model_selection import train_test_split

# Load your audio file
audio_file = '281474976883728_f1099_chunk_0.wav'  # Ensure this path is correct
y, sr = librosa.load(audio_file, sr=None)

# Extract MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
print("MFCCs shape:", mfccs.shape)

# Reshape MFCCs for model input (if necessary)
X = mfccs.T  # Transpose to shape (num_samples, num_features)

# Sample labels for demonstration
# (You should replace this with your actual labels)
y_labels = np.array([0] * (X.shape[0] // 2) + [1] * (X.shape[0] // 2))

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y_labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Print shapes
print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Testing set shape:", X_test.shape)
