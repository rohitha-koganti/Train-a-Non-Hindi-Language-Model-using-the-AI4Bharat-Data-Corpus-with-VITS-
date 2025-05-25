import librosa
import soundfile as sf

# Load and resample the audio to 22050 Hz
audio, sr = librosa.load("281474976883728_f1099_chunk_0.wav", sr=None)  # Load with original rate
audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=22050)  # Resample to 22050 Hz

# Save the resampled audio
sf.write("resampled_audio.wav", audio_resampled, 22050)
