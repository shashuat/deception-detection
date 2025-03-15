import whisper
import os
import sys
import json
import tqdm
import torch

from transformers import AutoModel, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from setup import get_env

ENV = get_env()
audio_dir = ENV["AUDIO_PATH"]
data_path = ENV["DATA_PATH"]
output_dir = ENV["TRANSCRIPTS_PATH"]

# load models
model = whisper.load_model("large")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

def get_padded_bert_embeddings(text, size=256, bert_embedding=768) -> torch.Tensor:
    tokens = bert_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=size)
    with torch.no_grad():
        outputs = bert_model(**tokens)

    embeddings = outputs.last_hidden_state  # (1, seq_len, 768)

    seq_len = embeddings.shape[1]
    if seq_len < size:
        pad_size = size - seq_len
        pad_tensor = torch.zeros((1, pad_size, bert_embedding))
        embeddings = torch.cat([embeddings, pad_tensor], dim=1)
    else:
        embeddings = embeddings[:, :size, :]

    return embeddings.squeeze(0)  # (size, bert_embedding)


def transcribe_audio(audio_path):
    # load and preprocess audio automatically to 16 kHz mono
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    result = model.transcribe(audio)
    return result

os.makedirs(output_dir, exist_ok=True)

audio_files = os.listdir(audio_dir)

# Example usage:
for filename in tqdm.tqdm(audio_files):
    audio_file = os.path.join(audio_dir, filename)
    output_path = os.path.join(output_dir, filename.replace(".wav", ".json"))

    output = json.load(open(output_path, 'r'))
    # output = transcribe_audio(audio_file)
    output["bert_embedding"] = get_padded_bert_embeddings(output["text"]).tolist()

    with open(output_path, 'w') as f: json.dump(output, f, indent=4)
