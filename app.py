import streamlit as st
import tensorflow as tf
import nltk
import json
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # Import transformers classes
nltk.download('punkt')  # Download NLTK data

# Load encoder model
encoder_model = tf.keras.models.load_model("encoder_model.h5", compile=False)

# Load decoder model
decoder_model = tf.keras.models.load_model("decoder_model.h5", compile=False)

# Load the third model
model = tf.keras.models.load_model("model.h5", compile=False)

# Load x_tokenizer
with open("x_tokenizer.json", "r", encoding="utf-8") as x_tokenizer_file:
    x_tokenizer_data = json.load(x_tokenizer_file)
x_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(x_tokenizer_data)

# Load y_tokenizer
with open("y_tokenizer.json", "r", encoding="utf-8") as y_tokenizer_file:
    y_tokenizer_data = json.load(y_tokenizer_file)
y_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(y_tokenizer_data)

# Function to generate a summary
def generate_summary(text):
    # Tokenize and preprocess the input text using NLTK tokenizer
    tokenized_text = word_tokenize(text)

    # Ensure the input sequence length matches the expected length (42)
    max_seq_length = 42
    if len(tokenized_text) > max_seq_length:
        tokenized_text = tokenized_text[:max_seq_length]
    else:
        # Pad the sequence if it's shorter than the expected length
        padding_length = max_seq_length - len(tokenized_text)
        tokenized_text.extend([''] * padding_length)

    # Encode the input text using the encoder model
    encoded_text = encoder_model.predict([tokenized_text])

    # Generate summary using the decoder model
    summary = decoder_model.predict(encoded_text)
    
    # Convert the tokenized input text to sequences
    input_sequence = x_tokenizer.texts_to_sequences([text])
    input_sequence = tf.keras.preprocessing.sequence.pad_sequences(input_sequence, maxlen=max_seq_length, padding="post")

    # Generate prediction using the third model
    generated_sequence = model.predict(input_sequence)
    generated_text = y_tokenizer.sequences_to_texts(generated_sequence)[0]
    
    return summary, generated_text

# Streamlit UI
st.title("Text Summarization App")

input_text = st.text_area("Enter your text:")
if st.button("Generate Summary"):
    if input_text:
        summary, generated_text = generate_summary(input_text)
        st.subheader("Generated Summary:")
        st.write(summary)
        st.subheader("Generated Text:")
        st.write(generated_text)
    else:
        st.warning("Please enter some text.")
