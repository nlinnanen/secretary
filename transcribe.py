from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# List all audio files in the folder
audio_files = [f for f in os.listdir() if f.endswith(".wav")]

# Transcribe each file
#for file_name in audio_files:
audio_file= open("speaker_SPEAKER_06_segment_52.wav", "rb")
transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file
)

print(transcription.text)
