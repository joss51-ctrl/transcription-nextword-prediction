import streamlit as st
import torch
import torch.nn.functional as F
import soundfile as sf
import io
import torchaudio
import re
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)

@st.cache_resource
def load_models():
    whisper_processor = WhisperProcessor.from_pretrained("whisper-model")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("whisper-model")
    gpt_tokenizer = AutoTokenizer.from_pretrained("gpt-model")
    gpt_model = AutoModelForCausalLM.from_pretrained("gpt-model")
    return whisper_processor, whisper_model, gpt_tokenizer, gpt_model

def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def is_alpha_only(text):
    clean = re.sub(r'[^\w\s]', '', text)
    return clean.isalpha()

def split_words(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

whisper_processor, whisper_model, gpt_tokenizer, gpt_model = load_models()

st.title("English Transcription")

uploaded_file = st.file_uploader("Upload audio (wav, mp3, m4a, flac)", type=["wav","mp3","m4a","flac"])

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes)

    audio_input, sr = sf.read(io.BytesIO(audio_bytes))
    if sr != 16000:
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_input = resampler(waveform).squeeze().numpy()

    input_features = whisper_processor(
        audio_input,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features

    forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language="en", task="transcribe")

    whisper_model.config.forced_decoder_ids = None
    whisper_model.generation_config.forced_decoder_ids = None

    with torch.no_grad():
        outputs = whisper_model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=256,
        )

    predicted_ids = outputs.sequences
    scores = outputs.scores

    logits = torch.stack(scores, dim=1)[0] 
    entropy = compute_entropy(logits)

    entropy_threshold = 4.0

    token_ids = predicted_ids[0]
    transcription = whisper_processor.decode(token_ids, skip_special_tokens=True)

    words = split_words(transcription)

    high_entropy_indices = []
    for i, ent in enumerate(entropy):
        if ent > entropy_threshold and i > 0:
            high_entropy_indices.append(i)

    highlighted_text = ""
    for i, w in enumerate(words):
        highlighted_text += w + " "

    st.subheader("Full Transcription")
    st.markdown(highlighted_text.strip(), unsafe_allow_html=True)


    suggestions = []
    max_suggestions = 3
    max_trials = 5

    for i in high_entropy_indices:
        context_ids = token_ids[max(0, i - 5):i]
        context_text = whisper_processor.tokenizer.decode(context_ids, skip_special_tokens=True).strip()

        if not context_text:
            continue

        encoding = gpt_tokenizer(
            context_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_ids = encoding["input_ids"].to(gpt_model.device)
        attention_mask = encoding["attention_mask"].to(gpt_model.device)

        with torch.no_grad():
            outputs_gpt = gpt_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5,
                num_return_sequences=1,
                do_sample=True,
                top_k=40,
                temperature=0.8,
                pad_token_id=gpt_tokenizer.eos_token_id
            )

        generated_text = gpt_tokenizer.decode(outputs_gpt[0], skip_special_tokens=True)
        generated_part = generated_text[len(context_text):].strip()
        generated_words = split_words(generated_part)

        suggested_words = []
        for word in generated_words:
            if is_alpha_only(word) and word not in suggested_words:
                suggested_words.append(word)
            if len(suggested_words) >= max_suggestions:
                break

        try:
            original_word = whisper_processor.tokenizer.decode([token_ids[i]], skip_special_tokens=True).strip()
        except:
            original_word = "<unk>"

        context_display_tokens = token_ids[max(0, i - 3):i + 2]
        context_display = whisper_processor.tokenizer.decode(context_display_tokens, skip_special_tokens=True).strip()

        suggestions.append((original_word, context_display, suggested_words))


    if high_entropy_indices:
        if suggestions:
            st.subheader("Replacement Suggestions for Low Confidence Words")
            for low_word, context, suggs in suggestions:
                highlighted_context = context.replace(
                    low_word, f'<span style="color:red; font-weight:bold;">{low_word}</span>', 1
                )
                st.markdown(
                    f"**Suggestion:** {highlighted_context}  →  _(suggested replacements: **{', '.join(suggs)}**)_",
                    unsafe_allow_html=True
                )

        st.warning("Some parts have high uncertainty.")
    else:
        st.success("No high uncertainty parts detected.")

