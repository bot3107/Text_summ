import streamlit as st
import tensorflow as tf
# from transformers import BertTokenizer
from transformers import BertTokenizer


# Load the text summarization model
model = tf.keras.models.load_model("text_summarization_model.h5", compile=False)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Function to generate a summary
def generate_summary(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="tf")
    input_ids = inputs["input_ids"]
    
    summary_ids = model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

        
import streamlit as st
from transformers import pipeline
from transformers import BertTokenizer as BertTokenizer


# Load the summarization pipeline
summarizer = pipeline("summarization")

# Streamlit UI
st.title("Text Summarization App")

input_text = st.text_area("Enter your text:")
if st.button("Generate Summary"):
    if input_text:
        summary = summarizer(input_text, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)[0]['summary_text']
        st.subheader("Generated Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text.")
