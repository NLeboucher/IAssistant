import os
import time
from google.cloud import speech
os.environ['GOOGLE_APPLICATION_CREDENTIALS']='IAssistant/Speech_to_Text/keyservicenicoadmin.json'
speech_client=speech.SpeechClient()
#speech_client=speech.SpeechAsyncClient
#transcribe Local Medial file
#file size : <10mb Len<1min
medial_file_name_mp3='IAssistant/Speech_to_Text/audiofiles/test.mp3'
medial_file_name_wav='IAssistant/Speech_to_Text/audiofiles/test.wav'

with open(medial_file_name_mp3,'rb') as f1:
    byte_data_mp3=f1.read()
audio_mp3=speech.RecognitionAudio(content=byte_data_mp3)

with open(medial_file_name_wav,'rb') as f2:
    byte_data_wav=f2.read()
audio_wav=speech.RecognitionAudio(content=byte_data_wav)  
##config files
config_mp3=speech.RecognitionConfig(
    sample_rate_hertz=48000,
    enable_automatic_punctuation=True,
    language_code='en-US',

)
config_wav=speech.RecognitionConfig(
    sample_rate_hertz=48000,
    enable_automatic_punctuation=True,
    language_code='en-US',
    audio_channel_count=1,

)
##transcribing the recognitionaudio objects
a=time.time()
repsponse_standard_mp3=speech_client.recognize(
    config=config_mp3,
    audio=audio_mp3,

)
print(f"{time.time()-a}: {repsponse_standard_mp3.results[0].alternatives[0].transcript},{repsponse_standard_mp3.results[0].alternatives[0].confidence}")
a=time.time()
repsponse_standard_wav=speech_client.recognize(
    config=config_wav,
    audio=audio_wav,

)
print(f"{time.time()-a}: {repsponse_standard_wav.results[0].alternatives[0].transcript},{repsponse_standard_wav.results[0].alternatives[0].confidence}")