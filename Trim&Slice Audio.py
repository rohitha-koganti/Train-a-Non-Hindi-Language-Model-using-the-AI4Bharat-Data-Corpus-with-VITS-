import librosa
import soundfile as sf

# Load audio and trim silence
audio, sr = librosa.load("normalized_audio.wav", sr=22050)
trimmed_audio, _ = librosa.effects.trim(audio, top_db=20)  # Remove silence with a threshold

# Save trimmed audio
sf.write("trimmed_audio.wav", trimmed_audio, sr)
