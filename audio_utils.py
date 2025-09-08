import streamlit as st
import numpy as np
from pydub import AudioSegment

def add_noise_to_audio(input_path, output_path, noise_level=0.005):
    try:
        sound = AudioSegment.from_wav(input_path)
        samples = np.array(sound.get_array_of_samples())
        noise = np.random.normal(0, sound.max * noise_level, len(samples))
        noisy_samples = samples + noise
        noisy_samples = np.clip(noisy_samples, -sound.max, sound.max).astype(samples.dtype)
        noisy_sound = sound._spawn(noisy_samples.tobytes())
        noisy_sound.export(output_path, format="wav")
        return True
    except Exception as e:
        st.error(f"Error adding noise: {e}")
        return False