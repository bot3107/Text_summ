import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Function to generate a summary
def generate_summary(text):
    input_text = "summarize: " + text  # Modify as per the model's requirement
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit UI
st.title("Text Summarization App")

input_text = st.text_area("Enter your text:")
if st.button("Generate Summary"):
    if input_text:
        summary = generate_summary(input_text)
        st.subheader("Generated Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text.")