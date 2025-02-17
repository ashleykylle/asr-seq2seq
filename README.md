# ABOUT
A seq2seq model for ASR in Filipino using a transformer-based model Whisper.

## Libraries Used
Project Dependencies:
- pytorch 
- torchaudio
- transformers (preload Whisper model)
- librosa
- jiwer (WER)

# Installations
pip install torch torchaudio transformers librosa jiwer

# Run
python model.py

# PROBLEMS TO NOTE:
- long audio is difficult for the code to process