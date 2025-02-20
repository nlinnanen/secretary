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
with ProgressHook() as hook:
    diarization = pipeline(FILE_NAME, hook=hook)

audio = AudioSegment.from_mp3(FILE_NAME)

# print the result
for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
    segment = audio[int(turn.start * 1000): int(turn.end * 1000)]
    segment.export(f"speaker_{speaker}_segment_{i}.wav", format="wav")


