
import streamlit as st
import torch
from transformers import AutoTokenizer
from transformers import pipeline
import openai as ai
from PyPDF2 import PdfReader


# Show title and description.
st.title("Cover letter Agent")

# Set Tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

# Load the text generation model from Hugging Face
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B', tokenizer = tokenizer)

# CV input
res_format = st.radio(
    "Do you want to upload or paste your CV",
    ('Upload', 'Paste'))
if res_format == 'Upload':
    res_file = st.file_uploader(' Upload your CV in pdf format')
    if res_file:
        pdf_reader = PdfReader(res_file)
        res_text = ""
        for page in pdf_reader.pages:
            res_text += page.extract_text()
else:
    res_text = st.text_input('Pasted CV elements')

#Additional CV information
with st.form('input_form'):
    job_desc = st.text_input('Pasted job description')
    user_name = st.text_input('Your name')
    company = st.text_input('Company name')
    manager = st.text_input('Hiring manager')
    role = st.text_input('Job title/role')
    referral = st.text_input('How did you find out about this opportunity?')
    ai_temp = st.number_input('AI Temperature (0.0-1.0) Input how creative the API can be', value=0.99)
    submitted = st.form_submit_button("Generate Cover Letter")

#Set model
if submitted:
    prompt = [
                {"role": "user", "content" : f"You will need to generate a cover letter based on specific resume and a job description"},
                {"role": "user", "content" : f"My resume text: {res_text}"},
                {"role": "user", "content" : f"The job description is: {job_desc}"},
                {"role": "user", "content" : f"The candidate's name to include on the cover letter: {user_name}"},
                {"role": "user", "content" : f"The job title/role : {role}"},
                {"role": "user", "content" : f"The hiring manager is: {manager}"},
                {"role": "user", "content" : f"How you heard about the opportunity: {referral}"},
                {"role": "user", "content" : f"The company to which you are generating the cover letter for: {company}"},
                {"role": "user", "content" : f"The cover letter should have three content paragraphs"},
                {"role": "user", "content" : f""" 
                In the first paragraph focus on the following: you will convey who you are, what position you are interested in, and where you heard
                about it, and summarize what you have to offer based on the above resume
                """},
                    {"role": "user", "content" : f""" 
                In the second paragraph focus on why the candidate is a great fit drawing parallels between the experience included in the resume 
                and the qualifications on the job description.
                """},
                        {"role": "user", "content" : f""" 
                In the 3RD PARAGRAPH: Conclusion
                Restate your interest in the organization and/or job and summarize what you have to offer and thank the reader for their time and consideration.
                """},
                {"role": "user", "content" : f""" 
                note that contact information may be found in the included resume text and use and/or summarize specific resume context for the letter
                    """},
                {"role": "user", "content" : f"Use {user_name} as the candidate"},
                
                {"role": "user", "content" : f"Generate a specific cover letter based on the above. Generate the response and include appropriate spacing between the paragraph text"}
            ]
    
    response_out = generator(prompt, max_length=2000, num_return_sequences=1)[0]['generated_text']
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    
    st.write(response_out)


    # Download a txt file
    st.download_button('Download', response_out)