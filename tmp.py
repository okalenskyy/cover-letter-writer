import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os

# Display instructions to the user
st.title("AI-Powered Cover Letter Generator")
st.write("Fill in the details below to generate a customized, professional cover letter.")

# Create input fields
textbox_company = st.text_input("Enter Company Name:")
textbox_job = st.text_input("Enter Job Title:")
textbox_resume = st.text_area("Enter Your Resume Summary:")

# Hugging Face authentication
HUGGING_FACE_TOKEN = "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"  # Replace with your actual Hugging Face token
os.environ["HUGGING_FACE_HUB_TOKEN"] = HUGGING_FACE_TOKEN

# Load the summarization and text generation models
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
   
except Exception as e:
    st.error(f"Error loading models: {e}")

# Helper function for chunking
def chunk_text(text, max_tokens, tokenizer):
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    return [tokens[i: i + max_tokens] for i in range(0, len(tokens), max_tokens)]

# Generate cover letter
if st.button("Generate Cover Letter"):
    if not textbox_company or not textbox_job or not textbox_resume:
        st.warning("Please fill in all fields before generating a cover letter.")
    else:
        with st.spinner("Generating your cover letter..."):
            try:
                # Tokenizer for splitting into manageable chunks
                tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

                # Input preparation
                ARTICLE = f"I am applying for the position of {textbox_job} at {textbox_company}. {textbox_resume}"

                # Chunk input text if it's too long
                max_input_tokens = 1024  # BART's token limit for input and output
                chunks = chunk_text(ARTICLE, max_input_tokens - 200, tokenizer)

                # Summarize each chunk and combine
                summarized_resume = ""
                for chunk in chunks:
                    chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
                    summarized = summarizer(chunk_text, max_length=200, min_length=50, do_sample=False)
                    summarized_resume += summarized[0]["summary_text"] + " "


                # Display the result
                st.subheader("Generated Cover Letter")
                st.write(summarized_resume.strip())
            except Exception as e:
                st.error(f"Error generating cover letter: {e}")