import pyaudio
import wave
import time
import os
from google.cloud import speech

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'IAssistant/Speech_to_Text/keyservicenicoadmin.json'
speech_client = speech.SpeechClient()

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "file.wav"

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

config = speech.RecognitionConfig(
    sample_rate_hertz=48000,
    enable_automatic_punctuation=True,
    language_code='en-US',
    audio_channel_count=1,
)

print("Recording audio, press Ctrl+C to stop...")
try:
    while True:
        print("Recording...")
        frames = []
        for i in range(0, int(RATE / CHUNK * 2)):
            data = stream.read(CHUNK)
            frames.append(data)

        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        
        with open(WAVE_OUTPUT_FILENAME, 'rb') as f:
            byte_data = f.read()
        audio_data = speech.RecognitionAudio(content=byte_data)

        response = speech_client.recognize(
            config=config,
            audio=audio_data,
        )
        try:
            print(f"Transcription: {response.results[0].alternatives[0].transcript}")
        except IndexError:
            print("\"silence\"")
except KeyboardInterrupt:
    pass
stream.stop_stream()
stream.close()
audio.terminate()