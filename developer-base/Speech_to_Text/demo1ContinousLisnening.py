import pyaudio
import wave
import argparse
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "file.wav"
 
audio = pyaudio.PyAudio()
def transcribe_streaming(stream_file):
    """Streams transcription of the given audio file."""
    import io
    from google.cloud import speech

    client = speech.SpeechClient()

    with io.open(stream_file, "rb") as audio_file:
        content = audio_file.read()

    # In practice, stream should be a generator yielding chunks of audio data.
    stream = [content]

    requests = (
        speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in stream
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    streaming_config = speech.StreamingRecognitionConfig(config=config)

    # streaming_recognize returns a generator.
    responses = client.streaming_recognize(
        config=streaming_config,
        requests=requests,
    )

    for response in responses:
        # Once the transcription has settled, the first result will contain the
        # is_final result. The other results will be for subsequent portions of
        # the audio.
        for result in response.results:
            print("Finished: {}".format(result.is_final))
            print("Stability: {}".format(result.stability))
            alternatives = result.alternatives
            # The alternatives are ordered from most likely to least.
            for alternative in alternatives:
                print("Confidence: {}".format(alternative.confidence))
                print(u"Transcript: {}".format(alternative.transcript))

# [END speech_transcribe_streaming]


if __name__ == "__main__":

    

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stream", help="File to stream to the API")
    args = parser.parse_args()
    args.stream=audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)



    
    transcribe_streaming(args.stream)