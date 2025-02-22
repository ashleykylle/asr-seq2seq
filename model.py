import re
import os
import torch
import torchaudio
import torchaudio.transforms as T
import jiwer
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# USE : whisper model medium for filipino
model_name = "openai/whisper-medium" 
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using : {device}")

# ! FILE NAME !
# ! FILE NAME !
# ! FILE NAME !
wav_file = "rec_001_7_fil.wav" # CHANGE TO FILE NAME 
file_path = os.path.join("WAV files", wav_file)

if not os.path.isfile(file_path):
    print(f"error with {file_path} path")
    exit()

dictionary = {
    # KNOWN ISSUES :
    # - "vitamin c" is being predicted as just "vitamins"
    # - "growee" is being predicted as "growing"


    # GENERAL WORDS
    "ayoko": "ayaw ko",
    "ganon": "ganoon",
    "di": "hindi",
    "eto": "ito",
    "yan": "iyan",
    "oh oh": "oo",
    # MEDICAL TERMS
    "alkasigay": "alka c",
    "apetazone": "appetason",
    "cilin": "ceelin",
    "celine": "ceelin",                       # UNSURE, THIS CAN CONFLICT WITH CELINE NAME IF EVER
    "sili chua balls": "ceelin chewables",
    "silituables": "ceelin chewables",
    "celine plus": "ceelin plus",
    "sherifur": "cherifer",
    "sherifir": "cherifer",
    "sherryfur": "cherifer",
    "sherifere": "cherifer",
    "sherry for": "cherifer",
    "fernsea": "fern-c",
    "gumis": "gummies",
    "growi": "growee",
    "lysine ion": "lysine iron",
    "nutrilene": "nutrilyn",
    "nutriplegs": "nutriplex",
    "novacy": "novacee",
    "nova sea": "novacee",
    "pet sink": "pedzinc",
    "price kids": "pryce kids",
    "scots": "scotts",
    "lamin a": "vitamin a",
}

def replace_equivalent_words(text):
    for key, value in sorted(dictionary.items(), key=lambda x: -len(x[0])):
        pattern = rf"\b{re.escape(key)}\b"
        text = re.sub(pattern, value, text, flags=re.IGNORECASE)
    return text

# text normalization
def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

# load audio and chunk for long audio
def split_audio(file_path, chunk_duration=5):
    waveform, sr = torchaudio.load(file_path)
    resampler = T.Resample(orig_freq=sr, new_freq=16000)
    waveform = resampler(waveform)

    samples_per_chunk = 16000 * chunk_duration
    chunks = [waveform[:, i:i + samples_per_chunk] for i in range(0, waveform.shape[1], samples_per_chunk)]
    
    return chunks

chunks = split_audio(file_path)
full_transcript = []

for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i+1}/{len(chunks)}...")

    inputs = processor(chunk.squeeze().numpy(), return_tensors="pt", sampling_rate=16000)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # force model to tagalog
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="tagalog", task="transcribe")

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs["input_features"],
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=444  # longer transcriptions
        )

    transcribed_chunk = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    full_transcript.append(transcribed_chunk)

final_transcript = " ".join(full_transcript)
print(f"\nTranscription: {final_transcript}\n")

# FOR LATER ON : compute WER
# TO DO : make transcripts for finetuning
# if we have reference transcripts, compare with predicted outputs
# FOR NOW HARD CODE REFERENCE TEXT
reference_text = "Hindi ko matandaan. Vitamin C. Pedzinc. Gummies. I forgot. Ceelin Plus. Growee. Ceelin. Vitamin C. Propan TLC. Ceelin chewables. Vitamin A. Nutriplex. Growee. Cherifer. Apetazone. Novacee. Fern-C. Pryze Kids. Celermin. Tableta. Ceelin Plus. B Complex. Lysine Iron. Neurolem. Nutriplex. Cherifer. Immunomax. Alka C. Nutrilyn. Scotts."
normalized_reference = normalize_text(reference_text)
normalized_predicted = normalize_text(final_transcript)

final_reference = replace_equivalent_words(normalized_reference)
final_predicted = replace_equivalent_words(normalized_predicted)

wer = jiwer.wer(final_reference, final_predicted)

print(f"Reference: {final_reference}")
print(f"Predicted: {final_predicted}")
print(f"WER: {wer:.2%}")

# !!! BEFORE RUNNING !!!
# make sure you're in the correct directory
# make sure the file name is correct
# WER may be a bit off due to some punctuations and capitalization issues