import yt_dlp
import csv
import os
from pydub import AudioSegment
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.environ.get("HUGGINGFACE_TOKEN", None))

# send pipeline to GPU (when available)
pipeline.to(torch.device("mps"))


def download_audio_if_not_exists(url, file_name):
    if os.path.exists(file_name):
        return
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
        'outtmpl': file_name.replace(".mp3", "")  # Output file name
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def split_audio_by_agenda_points_if_not_exists(audio_file, start_time, end_time, agenda_point):
    print(f"Splitting {audio_file} from {start_time} to {end_time}")
    output_dir = "./audio/agenda_points"
    output_path = os.path.join(output_dir, f"ap_{agenda_point}_{start_time}.mp3")
    if os.path.exists(output_path):
        print(f"File {output_path} already exists")
        return output_path
    audio = AudioSegment.from_mp3(audio_file)
    segment = audio[int(start_time)*1000: int(end_time)*1000]
    segment.export(output_path, format="mp3")
    return output_path


def split_audio_by_speaker(file_name):
    with ProgressHook() as hook:
        diarization = pipeline(file_name, hook=hook)
    print(f"Splitting {file_name}")
    audio = AudioSegment.from_mp3(file_name)
    segments = []
    for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        if turn.duration < 1:
            continue
        segment = audio[int(turn.start * 1000): int(turn.end * 1000)]
        segment_name = f"./audio/speakers/{file_name.replace('./audio/agenda_points/', '').replace('.mp3', '')}_speaker_{speaker}_segment_{i}.wav"
        segment.export(segment_name, format="wav")
        segments.append({"path": segment_name, "segment": i, "speaker": speaker, "start": turn.start, "end": turn.end})

    return segments


def transcribe_audio(file_name):
    audio_file = open(file_name, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

    return transcription.text

def seconds_to_hms(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{hours}h{minutes}m{seconds}s"

def write_to_csv(transcription, agenda_point, segment):
    with open("./transcriptions.csv", mode="a", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        start_of_segment = seconds_to_hms(float(agenda_point['start_time']) + float(segment['start']))
        link_with_time = f"{agenda_point['url']}&t={start_of_segment}"
        writer.writerow([agenda_point['agenda_point'],segment['speaker'],start_of_segment,link_with_time, transcription])


def process_csv(csv_file):
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        agenda_point_files = []
        for row in reader:
            file_name = f"./audio/youtube/{row['url'].split('=')[-1]}.mp3"
            download_audio_if_not_exists(
                row['url'], file_name)
            print(f"Downloaded {file_name}")
            path = split_audio_by_agenda_points_if_not_exists(
                file_name, row['start_time'], row['end_time'], row['agenda_point'])
            agenda_point_files.append({"path": path, "row": row})
            

    for entry in agenda_point_files:
        file_name = entry['path']
        row = entry['row']
        segments = split_audio_by_speaker(file_name)
        for segment in segments:
            transcription = transcribe_audio(segment['path'])
            write_to_csv(transcription, row, segment)


if __name__ == "__main__":
    csv_file = "letsgo.csv"  # CSV file name
    # create the folders
    os.makedirs("./audio/youtube", exist_ok=True)
    os.makedirs("./audio/speakers", exist_ok=True)
    os.makedirs("./audio/agenda_points", exist_ok=True)

    process_csv(csv_file)
