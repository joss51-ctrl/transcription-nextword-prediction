# Enhancing Transcription Accuracy with Next-Word Prediction
One of the main challenges in transcription systems is achieving high accuracy, especially when dealing with low-quality audio, diverse accents, or complex vocabularies. To address this issue, we integrate Next Word Prediction into the transcription pipeline. By combining speech-to-text and next-word prediction, the system not only converts audio into text but also leverages conversational flow and context to produce more accurate and context-aware transcriptions.

# Features
* Speech-to-Text with Whisper – Transcribe audio into text.
  
* Next Word Prediction – Predicts the next word based on context.
  
* Entropy-based Confidence – Calculates entropy to determine prediction confidence.

* Context-Aware Transcription – Enhances transcription accuracy with word suggestions.

# Models
* Models: GPT-Small, GPT-Neo-125M

* Dataset: DailyDialog (multi-turn dialogue)

* Preprocessing: Dialogues segmented using <eou> (end of utterance) token. Added as a special token in GPT2Tokenizer.

* Result: GPT-Small performs slightly better than GPT-Neo-125M; therefore, GPT-Small is used in the final app

# Installation
git clone https://github.com/your-username/transcription-nextword-prediction.git
cd transcription-nextword-prediction

# Install dependencies
pip install -r requirements.txt

# Running the Application
streamlit run app.py
  
