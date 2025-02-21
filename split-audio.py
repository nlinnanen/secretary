from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pydub import AudioSegment
import torch
import os

FILE_NAME = "trimtrim.mp3"

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.environ.get("HUGGINGFACE_TOKEN", None))

# send pipeline to GPU (when available)
pipeline.to(torch.device("mps"))

# apply pretrained pipeline (with optional progress hook)
# List all audio files in the folder
audio_files = [f"./audio/{f}" for f in os.listdir("./audio/") if f.endswith(".mp3")]

# Transcribe each file
for file_name in audio_files:
  with ProgressHook() as hook:
      diarization = pipeline(file_name, hook=hook)

  audio = AudioSegment.from_mp3(file_name)

  # print the result
  for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
      if turn.duration < 0.5:
          continue
      segment = audio[int(turn.start * 1000): int(turn.end * 1000)]
      segment.export(f"./audio/splitted/{file_name.replace("./audio/", "")}_speaker_{speaker}_segment_{i}.wav", format="wav")


