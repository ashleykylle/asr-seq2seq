import re
import os
import csv
import torch
import torchaudio
import torchaudio.transforms as T
import jiwer
import random
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# USE : whisper model medium for Filipino
model_name = "openai/whisper-medium"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using : {device}")

# Define paths
root_folder = "D:\\Schoolshit\\Thesis\\dataset - Copy"
csv_file = "transcription.csv"

def load_transcriptions(csv_file):
    transcriptions = []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if len(row) == 2:
                transcriptions.append(row[1])
    return transcriptions

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

transcriptions = load_transcriptions(csv_file)

wav_files = []
for entry in sorted(os.scandir(root_folder), key=lambda e: e.name):
    if entry.is_dir():
        folder_path = entry.path
        for file in sorted(os.listdir(folder_path)):
            if file.endswith(".wav"):
                wav_files.append(os.path.join(folder_path, file))

# Train-test split (90% train, 10% validation)
wav_train, wav_valid, trans_train, trans_valid = train_test_split(
    wav_files, transcriptions, test_size=0.1
)

def evaluate_dataset(wav_files, transcriptions, dataset_name):
    print(f"\nEvaluating {dataset_name} set...")

    total_wer_1_word = 0
    total_wer_2_word = 0
    total_wer_3_or_more_words = 0

    count_1_word = 0
    count_2_word = 0
    count_3_or_more_words = 0

    file_count = 0
    total_wer = 0

    for i, file_path in enumerate(wav_files):
        print(f"Processing {file_path}...")
        waveform, sr = torchaudio.load(file_path)
        resampler = T.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        
        inputs = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="tagalog", task="transcribe")
        
        with torch.no_grad():
            predicted_ids = model.generate(
                inputs["input_features"],
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=444
            )
        
        predicted_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Normalize texts before computing WER
        predicted_text = normalize_text(predicted_text)
        reference_text = normalize_text(transcriptions[i])
        
        wer = jiwer.wer(predicted_text, reference_text) * 100

        # Count words in the reference text
        word_count = len(reference_text.split())

        if word_count == 1:
            total_wer_1_word += wer
            count_1_word += 1
        elif word_count == 2:
            total_wer_2_word += wer
            count_2_word += 1
        else:
            total_wer_3_or_more_words += wer
            count_3_or_more_words += 1

        total_wer += wer
        file_count += 1

        print(f"Reference: {reference_text}")
        print(f"Predicted: {predicted_text}")
        print(f"WER (File): {wer:.2f}%\n")

    # Final WER Statistics
    if count_1_word > 0:
        final_wer_1_word = total_wer_1_word / count_1_word
        print(f"\nFinal WER for 1-word samples: {final_wer_1_word:.2f}% ({count_1_word} samples)")

    if count_2_word > 0:
        final_wer_2_word = total_wer_2_word / count_2_word
        print(f"Final WER for 2-word samples: {final_wer_2_word:.2f}% ({count_2_word} samples)")

    if count_3_or_more_words > 0:
        final_wer_3_or_more_words = total_wer_3_or_more_words / count_3_or_more_words
        print(f"Final WER for 3 or more words: {final_wer_3_or_more_words:.2f}% ({count_3_or_more_words} samples)")
    
    # Final WER Statistics
    if file_count > 0:
        final_wer = total_wer / file_count
        print(f"\nFinal Overall WER: {final_wer:.2f}%")
    
    print(f"\nTotal files processed: {file_count}")

evaluate_dataset(wav_valid, trans_valid, "Validation")
