import io
import librosa
import numpy as np
from scipy.signal import find_peaks
import soundfile as sf
import speech_recognition as sr
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def decode_audio(audio_file):
    """Decode the uploaded audio file and return the audio signal and sample rate."""
    try:
        with open(audio_file,"rb") as audio_file:
            audio_io = io.BytesIO(audio_file.read())
        y, sample_rate = librosa.load(audio_io, sr=None)
        return y, sample_rate
    except Exception as e:
        logging.error(f"Error decoding audio: {e}")
        raise ValueError("Invalid audio file.")

def check_liveliness(y, sample_rate):
    """Check for liveliness by analyzing the audio's envelope and peak variations."""
    try:
        envelope = np.abs(librosa.onset.onset_strength(y=y, sr=sample_rate))
        peaks, _ = find_peaks(envelope, height=0.1)
        peak_count = len(peaks)
        is_lively = peak_count > 10
        return is_lively
    except Exception as e:
        logging.error(f"Error checking liveliness: {e}")
        raise ValueError("Liveliness check failed.")

def verify_number(number, y, sample_rate):
    """Verify if the spoken number is present in the audio."""
    recognizer = sr.Recognizer()
    recognized_text = ""
    try:
        with io.BytesIO() as audio_io:
            sf.write(audio_io, y, sample_rate, format='WAV')
            audio_io.seek(0)
            with sr.AudioFile(audio_io) as source:
                audio = recognizer.record(source)
        
        recognized_text = recognizer.recognize_google(audio)
        recognized_number = "".join(re.findall(r'\d+', recognized_text))
        logging.info(f"Recognized text: {recognized_number}")
        
        logging.info(f"Number in verify number: {number}")
        is_verified = number == recognized_number
        return is_verified, recognized_number
    except sr.UnknownValueError:
        logging.warning("Speech recognition could not understand the audio.")
        return False, recognized_text
    except sr.RequestError as e:
        logging.error(f"Google Speech Recognition request failed: {e}")
        return False, recognized_text
    except Exception as e:
        logging.error(f"Unexpected error during speech recognition: {e}")
        raise ValueError("Speech recognition failed.")

def process_audio(number, audio_file):
    """Main function to process the audio and verify the number and liveliness."""
    try:
        logging.info("Starting audio processing.")
        
        y, sample_rate = decode_audio(audio_file)
        
        is_lively = check_liveliness(y, sample_rate)
        
        is_number_verified, recognized_text = verify_number(number, y, sample_rate)

        result = {
            "liveliness": is_lively,
            "numberVerified": is_number_verified,
            "transcribedOTP": recognized_text,
            "actualOTP": number,
            "audioDuration": librosa.get_duration(y=y, sr=sample_rate)
        }
        
        logging.info("Audio processing completed successfully.")
        return result
    except Exception as e:
        logging.error(f"Error during audio processing: {e}")
        raise ValueError("Audio processing failed.")
