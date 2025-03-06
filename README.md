# Aautosecretary

A tool that downloads audio from a youtube video, splits it by speaker, and transcribes it. For diarization [pyannote-audio](https://github.com/pyannote/pyannote-audio?tab=readme-ov-file) is used and for transcription [OpenAI API](https://platform.openai.com/docs/guides/speech-to-text) is used.

## Usage

The script reads the data from data.csv file. The data.csv file should have atleast the following columns to work properly.

```csv
agenda_point,start_time,end_time,url
```

Agenda point is to differentiate between parts of the video. Start time and end time are the timestamps of the video transcription should happen. The url is the youtube video url.

## Installation

Activate your virtual environment and install the requirements.

```bash
python -m venv venv
source venv/bin/activate # for linux or mac
venv\Scripts\activate # for windows
pip install -r requirements.txt
```

Copy the `.env.example` file to `.env` and fill in the required fields.

```bash
cp .env.example .env
```

Get the OpenAI API key from the [OpenAI API](https://platform.openai.com/api-keys) and the Huggning Face API key from the [Hugging Face API](https://hf.co/settings/tokens) and fill in the `.env` file.

> **_NOTE:_**  Using the OpenAI API is not free. You will be charged for the usage. For me it was around 1.5$ for the whole CM.

Run the script
  
```bash
python script.py
```

