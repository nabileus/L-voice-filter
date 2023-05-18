import librosa
import soundfile as sf
import numpy as np

# Load the original audio file
audio, sr = librosa.load('input.wav', sr=None)

# Define the pitch shift factors for the two outputs
pitch_shift_factors = [3, -2]

# Initialize an array to store the mixed audio
mixed_audio = np.zeros_like(audio)

# Process each pitch shift factor
for i, pitch_shift_factor in enumerate(pitch_shift_factors):
    # Shift the pitch of the audio
    pitch_shifted_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift_factor, bins_per_octave=12)

    # Calculate the stretch factor based on durations
    original_duration = len(audio) / sr
    pitch_shifted_duration = len(pitch_shifted_audio) / sr
    stretch_factor = pitch_shifted_duration / original_duration

    # Stretch the pitch-shifted audio by repeating the samples
    stretched_audio = librosa.effects.time_stretch(pitch_shifted_audio, rate=stretch_factor)

    # Mix the stretched audio with the original audio
    mixed_audio = np.add(mixed_audio[:len(stretched_audio)], stretched_audio)  # Mix at the same length as the stretched audio

    # Save the stretched audio to a new file
    #output_filename = f'output_audio_{i+1}.wav'  # Create a unique filename for each output
    #sf.write(output_filename, stretched_audio, sr, format='wav')

# Mix the original audio with the final mixed audio
mixed_audio = np.add(mixed_audio[:len(audio)], audio)  # Mix at the same length as the original audio

# Save the mixed audio to a file
sf.write('output.wav', mixed_audio, sr, format='wav')
