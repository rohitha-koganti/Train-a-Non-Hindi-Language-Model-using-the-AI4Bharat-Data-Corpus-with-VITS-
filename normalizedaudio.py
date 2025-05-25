from pydub import AudioSegment

# Load audio and normalize volume
audio = AudioSegment.from_file("resampled_audio.wav")
normalized_audio = audio.apply_gain(-audio.max_dBFS)  # Normalize to 0 dBFS

# Save normalized audio
normalized_audio.export("normalized_audio.wav", format="wav")
