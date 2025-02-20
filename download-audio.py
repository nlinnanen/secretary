import yt_dlp
import csv
import os

def download_audio_with_timestamp(url, start_time, end_time, output_file):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
        'download_sections': { '*': [(start_time, end_time)] },
        'outtmpl': output_file  # Output file name
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def process_csv(csv_file):
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            url = row['url']
            start_time = row['start_time']  # Format: HH:MM:SS
            end_time = row['end_time']    # Format: HH:MM:SS
            output_filename = f"audio_{row['agenda_point']}.mp3"  # Unique filename based on ID
            
            print(f"Downloading: {url} from {start_time} to {end_time}...")
            download_audio_with_timestamp(url, start_time, end_time, output_filename)
            print(f"Saved as: {output_filename}\n")

if __name__ == "__main__":
    csv_file = "test.csv"  # CSV file name
    process_csv(csv_file)