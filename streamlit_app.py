import streamlit as st
import torch
import os
from transformers import AutoTokenizer
from transformers import pipeline
import openai as ai
from PyPDF2 import PdfReader

def chunkerize(text, max_tokens, tokenizer):
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    return [tokens[i: i + max_tokens] for i in range(0, len(tokens), max_tokens)]

# Show title and description.
st.title("Cover letter Agent")

# Hugging Face authentication
HUGGING_FACE_TOKEN = "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"  # Replace with your actual Hugging Face token
os.environ["HUGGING_FACE_HUB_TOKEN"] = HUGGING_FACE_TOKEN


# st.info("Please input your API key to continue.", icon="🗝️")
# os.environ["HUGGING_FACE_HUB_TOKEN"] = st.text_input("API Key", type="password")

# Load the text generation model from Hugging Face
# try: 
#     generator = pipeline("summarization", model="facebook/bart-large-cnn")
# except Exception as e:
#     st.error(f"Model can not be loaded. Error: {e}")

# CV input
# res_format = st.radio(
#     "Do you want to upload or paste your CV",
#     ('Upload', 'Paste'))
# if res_format == 'Upload':
#     res_file = st.file_uploader(' Upload your CV in pdf format')
#     if res_file:
#         pdf_reader = PdfReader(res_file)
#         res_text = ""
#         for page in pdf_reader.pages:
#             res_text += page.extract_text()
# else:
res_text = st.text_input('CV')

#Additional CV information
with st.form('input_form'):
    job_desc = st.text_input('Job description')
    user_name = st.text_input('Your name')
    company = st.text_input('Company name')
    manager = st.text_input('Hiring manager')
    role = st.text_input('Job title/role')
    referral = st.text_input('How did you find out about this opportunity?')
    ai_temp = st.number_input('AI Temperature (0.0-1.0) Input how creative the API can be', value=0.99)
    submitted = st.form_submit_button("Generate Cover Letter")

# ------------------------
# Generate cover letter
if submitted:
# if st.button("Generate Cover Letter"):
    with st.spinner("Writing..."):
        # Load the text generation model from Hugging Face
        try: 
            generator = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            st.error(f"Model can not be loaded. Error: {e}")

        try:
            # Tokenizer for splitting into manageable chunks
            tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
            # Input preparation
            ARTICLE = f"I am applying for the position of {job_desc} at {company}. {res_text}"

            # ARTICLE = f"You will need to generate a cover letter based on specific resume and a job description, My resume text: {res_text}, The job description is: {job_desc}, The candidate's name to include on the cover letter: {user_name}, The job title/role : {role}, The hiring manager is: {manager}, How you heard about the opportunity: {referral}, The company to which you are generating the cover letter for: {company}, The cover letter should have three content paragraphs. In the first paragraph focus on the following: you will convey who you are, what position you are interested in, and where you hear about it, and summarize what you have to offer based on the above resume, In the second paragraph focus on why the candidate is a great fit drawing parallels between the experience included in the resume and the qualifications on the job description. In the 3RD PARAGRAPH: Conclusion Restate your interest in the organization and/or job and summarize what you have to offer and thank the reader for their time and consideration. note that contact information may be found in the included resume text and use and/or summarize specific resume context for the letter Use {user_name} as the candidate, Generate a specific cover letter based on the above. Generate the response and include appropriate spacing between the paragraph text"

            # Chunk input text if it's too long
            max_input_tokens = 1024  # BART's token limit for input and output
            chunks = chunkerize(ARTICLE, max_input_tokens - 200, tokenizer)

            # Response in chunks
            response = ""
            for chunk in chunks:
                chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
                summarized = generator(chunk_text, max_length=500, min_length=50, do_sample=False)
                response += summarized[0]["summary_text"] + " "

            # Display the result
            st.subheader("Cover Letter")
            st.write(response.strip())
            
            # Download a txt file
            st.download_button('Download', response)
        except Exception as e:
            st.error(f"Error in generation of cover letter: {e}")


