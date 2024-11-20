# import streamlit as st
# from openai import OpenAI

# import json
# import requests
# import time
# import cv2

import streamlit as st
from transformers import pipeline
import openai as ai
from PyPDF2 import PdfReader


# Show title and description.
st.title("Cover letter Agent")
# st.write(
#      "To use this app, you need to provide an OpenAI API key, which you can get [here](https://). "
#  )

#1 Load the text generation model from Hugging Face
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')


#####--

# Set your OpenAI API key
# ai.api_key = st.text_input("Tocken", type="password")

#CV input
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
            {"role": "user", "content": f"You will need to generate a cover letter based on specific resume and a job description"},
            {"role": "user", "content": f"My resume text: {res_text}"},
            # ... (include other user messages)
        ]
    
    response_out = generator(prompt, max_length=2000, num_return_sequences=1)[0]['generated_text']
    st.write(response_out)



# if submitted:
#     completion = ai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         temperature=ai_temp,
#         messages=[
#             {"role": "user", "content": f"You will need to generate a cover letter based on specific resume and a job description"},
#             {"role": "user", "content": f"My resume text: {res_text}"},
#             # ... (include other user messages)
#         ]
#     )
#     response_out = completion['choices'][0]['message']['content']
#     st.write(response_out)

    # Download a txt file
    st.download_button('Download the cover_letter', response_out)





# if not ai.api_key:
#     st.info("Please add your tocken to continue.", icon="🗝️")
# else:
    
#     headers = {"Authorization": f"Bearer {token_access}"}
#     API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"

#     if uploaded_file and question:

#         # Process the uploaded file and question.
#         document = uploaded_file.read().decode()
#         messages = [
#             {
#                 "role": "user",
#                 "content": f"Here's a document: {document} \n\n---\n\n {question}",
#             }
#         ]

#         # Generate an answer using the OpenAI API.
#         stream = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=messages,
#             stream=True,
#         )

#         # Stream the response to the app using `st.write_stream`.
#         st.write_stream(stream)
