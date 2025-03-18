import streamlit as st
import google.generativeai as genai
import io
import os
import base64
import time
import pandas as pd
from io import StringIO
from PIL import Image
from pdf2image import convert_from_path
from dotenv import load_dotenv 

# Load environment variables
load_dotenv()

# Set up Google Gemini API
# GOOGLE_API_KEY = "AIzaSyDFXWld_m2z9pwDYHPMw3_wlM63NzB0lOE"     # "AIzaSyAfX2Zde2ml37t_CfgpryWxEjlmGOsrW4U"  # Replace with your API key
# Get API Key from .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Function to extract text using Gemini
def extract_text_from_image(image_path):
    # Choose a Gemini model.
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")    # gemini-pro-vision  gemini-1.5-pro gemini-1.5-flash gemini-2.0-flash

    prompt = '''
            Extract the text in the sales record image as a sales record csv including the date and the business name which is the title of the image.
            Final table format - business_name,product,quantity,unit_price,date(yyyy-mm-dd)
        '''
    
    # Prompt the model with text and the previously uploaded image.
    response = model.generate_content([image_path, prompt])
    if response.text:
        print("Extracted Text:")
        return response.text
    else:
        print("Failed to extract text from the image.")
    # return response.text

def final_action(images):
    # Perform OCR on each extracted image
    extracted_texts = []
    start_time = time.time()
    with st.spinner("Extracting text..."):
        for img in images:
        # for i in range(0, len(images), 3):
            extracted_texts.append(extract_text_from_image(img)[6:][:-4])
            # extracted_texts.append('Times...')
            print(f'>>>> Page: {images.index(img)+1} done')
            # print(extracted_texts)
            # time.sleep(2)  # Wait 2 seconds before the next request

    full_text = "\n\n".join(extracted_texts)

    try:
        df = pd.read_csv(StringIO(full_text))
        df = df[df['business_name'] != 'business_name']

        # Display DataFrame
        st.subheader("📊 Extracted Sales Data")
        st.dataframe(df)

        end_time = time.time()
        elapsed_time = end_time - start_time
        st.success(f"✅ Extraction Completed in {elapsed_time:.2f} seconds")

    except:
        st.error("Error: Failed to parse structured data. Check the extracted text.")


# # Streamlit UI
st.title("Kola Sales Report Extraction")
st.subheader("Upload a scanned sales report (Image or PDF)")

# File uploader
uploaded_file = st.file_uploader("Upload a file", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    file_type = uploaded_file.type
    
    images = []  # Store images extracted from PDF (if applicable)

    # Process Image Files
    if file_type in ["image/png", "image/jpeg", "image/jpg"]:
        image = Image.open(uploaded_file)
        images.append(image)
        for img in images:
                st.image(img, caption="Extracted Page", use_container_width=True)

        final_action(images)

    # Process PDF Files (Extract images from scanned PDFs)
    elif file_type == "application/pdf":
        pdf_path = "uploaded_pdf.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Get PDF metadata to determine total pages
        images = convert_from_path(pdf_path, first_page=1, last_page=1)
        total_pages = len(convert_from_path(pdf_path))

        # User input for page range
        start_page = st.number_input("Start Page", min_value=1, max_value=total_pages, value=1, step=1)
        end_page = st.number_input("End Page", min_value=1, max_value=total_pages, value=total_pages, step=1)

        if st.button("Process PDF"):
            st.write(f"Processing pages {start_page} to {end_page}...")

            # ⏱ Start Timer
            start_time = time.time()

            # Convert PDF pages to images
            images = convert_from_path(pdf_path, first_page=start_page, last_page=end_page)
            # extracted_text = ""

            for img in images:
                st.image(img, caption="Extracted Page", use_container_width=True)

            final_action(images)