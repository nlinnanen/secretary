from openai import OpenAI
import os
from dotenv import load_dotenv
import csv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# List all audio files in the folder
audio_files = [f for f in os.listdir(
    "./audio/splitted/") if f.endswith(".wav")]

# Transcribe each file
for file_name in audio_files:
    audio_file = open(f"./audio/splitted/{file_name}", "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

    with open("./transcriptions.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([file_name, transcription.text])
