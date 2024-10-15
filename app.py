# Imports 
import os
import openai
import requests
import pandas as pd
from fpdf import FPDF
from groq import Groq
import streamlit as st
from dotenv import load_dotenv
from pdfminer.high_level import extract_text

# Load environment variables
load_dotenv()

# Set up API keys
key = os.getenv('api_key')
client = Groq(api_key=key)
openai.api_key = os.getenv('openai_api_key')  

# Function to extract text from the uploaded PDF file
def extract_text_from_pdf(pdf_file):
    text = extract_text(pdf_file)
    return text

# Function to clean text using groq API
def get_clean_groqResponse(text):
    try:
        system_message = f"""
        You are an assistant that cleans and refines text coming from PDF. Please clean and structure the following content extracted from a PDF. Make the text more readable and coherent with proper markdown.
        """
        user_input_with_context = f"""I have extracted the following text from a PDF. Please clean and refine it, applying proper markdown formatting for readability.

        ## Extracted Text:

        {text}"""

        # Use Groq model to generate a response
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": user_input_with_context,
                }
            ],
            model="llama3-70b-8192",
        )

        # # Use OpenAI's GPT model to generate a response
        # chat_completion = openai.ChatCompletion.create(
        #     model="gpt-4",  # You can use 'gpt-4' or 'gpt-3.5-turbo'
        #     messages=[
        #         {"role": "system", "content": system_message},
        #         {"role": "user", "content": user_input_with_context}
        #     ]
        # )

        full_response = chat_completion.choices[0].message.content
        return full_response
    except Exception as e:
        return f"Error processing your request: {str(e)}"
    
# Function to generate a lesson plan from the edited text
def generate_lesson_plan(edited_text):
    try:
        system_message = f"""
        You are an assistant that generates well-structured lesson based on the provided content. 
        Create a lesson with the following sections:
        - Introduction: Introduction of the chapter.
        - Main Body: Core teaching content, including key concepts and supporting details.
        - Class Activity: A classroom activity to apply the lesson.
        """
        user_input_with_context = f"""I have the following content that needs to be turned into a lesson to teach to the students. Structure it into a lesson with an introduction, main body, and class activity.

        ## Content:

        {edited_text}
        
        ## Important Instructions ## 
        Generated lesson must we well-structured, informative, and engaging.
        Just create the lesson and do not attach extra text along with it.
        Each section must be well structured such that teacher could directly use it to teach 
        It must cover essential information from the chapter.
        Follow all these instructions strictly.
        """

        # chat_completion = client.chat.completions.create(
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": system_message,
        #         },
        #         {
        #             "role": "user",
        #             "content": user_input_with_context,
        #         }
        #     ],
        #     model="llama3-70b-8192",
        # )

        # Use OpenAI's GPT model to generate a response
        chat_completion = openai.ChatCompletion.create(
            model="gpt-4",  
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input_with_context}
            ]
        )

        lesson_plan_response = chat_completion.choices[0].message.content
        return lesson_plan_response
    except Exception as e:
        return f"Error generating the lesson plan: {str(e)}"
    
# Function to generate summary of the text
def get_summary_text(text, num):
    try:
        system_message = f"""You are an assistant that will receive a lesson plan. Your task is to create description for the image, that will be used to generate the image. The description should highlight key visual elements, settings, objects, and actions that are crucial for generating realistic and illustrative images related to the lesson content."""

        user_input_with_context = """You will receive the lesson plan and number of images to generate below, and your task is to create a description of each image(within 250 characters) that focuses on key visual elements and essential content of the lesson. The number of Images will tell you how many description you have to create. The output format will be a dictioary.

        ## Summary Text ##

        sumr_text
        
        ## Number of images ##
        
        num_img

        ## Output Format ##

        {'image_1':'description1', 'image_2':'description2', 'image_3':'description3'}
        
        ## Important Instructions ##
        Your description should not contain text that is not allowed by openai safety system to generate image.
        Just return the dictionary with description of the images and do not attach extra text along with it.
        Follow all the instruction strictly."""

        # Replace placeholders in user_prompt
        user_input_with_context = user_input_with_context.replace("sumr_text", text)
        user_input_with_context = user_input_with_context.replace("num_img", str(num))

        # chat_completion = client.chat.completions.create(
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": system_message,
        #         },
        #         {
        #             "role": "user",
        #             "content": user_input_with_context,
        #         }
        #     ],
        #     model="llama3-70b-8192",
        # )

        # Use OpenAI's GPT model to generate a response
        chat_completion = openai.ChatCompletion.create(
            model="gpt-4",  
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input_with_context}
            ],
            max_tokens=4096,
        )

        full_response = chat_completion.choices[0].message.content
        full_response = eval(full_response)
        # print("full_response: ", full_response)
        return full_response
    except Exception as e:
        return f"Error processing your request: {str(e)}"

# Function to create prompts for image generation from a dictionary
def create_image_prompts(image_descriptions):
    # print(type(image_descriptions))
    prompts = []
    for key, description in image_descriptions.items():
        prompt = f"""Create a realistic and educational image for the following description:
        {description}.
        
        ## Important Instructions ##
        Please ensure there is no text in any of the image.
        The image should visually represent the core concepts, relevant diagrams or charts that reflect the descriptions."""
        prompts.append(prompt)

    # print("Prompts: ", prompts)
    return prompts


# Function to generate multiple images using DALL-E based on the lesson plan
def generate_images(prompts):
    try:
        image_urls = []
        for prompt in prompts:
            response = openai.Image.create(
                model="dall-e-3",
                prompt=prompt,
                n=1,  # Only one image per request
                size="1024x1024",
                quality="hd"
            )
            # Collect the generated image URL
            image_url = response['data'][0]['url']
            image_urls.append(image_url)
        
        return image_urls
    except Exception as e:
        return f"Error generating the image: {str(e)}"



# Function to clean text by replacing unsupported characters for PDF
def clean_text_for_pdf(text):
    # Replace unsupported characters with equivalents
    replacements = {
        '\u2014': '-',  # Replace em dash with a simple dash
        '\u2013': '-',  # Replace en dash with a simple dash
        '’': "'",       # Replace curly apostrophe with straight apostrophe
        '“': '"',       # Replace opening quotation mark with straight quote
        '”': '"',       # Replace closing quotation mark with straight quote
    }
    
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    
    return text

# Function to generate a PDF containing the lesson plan and illustrations
def generate_pdf(lesson_plan, image_urls):
    # Clean the lesson plan text to remove unsupported characters
    lesson_plan = clean_text_for_pdf(lesson_plan)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Set title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, txt="Lesson Plan with Illustrations", ln=True, align='C')
    
    # Add lesson plan content
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, lesson_plan)
    
    # Add images
    for i, url in enumerate(image_urls):
        try:
            # Download the image from URL
            response = requests.get(url)
            img_path = f"C:\\Users\\Tanishq-AxlrevAi\\Documents\\Assignment\\images\\lesson_image_{i+1}.png"
            with open(img_path, 'wb') as img_file:
                img_file.write(response.content)
            
            pdf.add_page()
            pdf.cell(200, 10, txt=f"Illustration {i+1}", ln=True, align='C')
            pdf.image(img_path, x=10, y=30, w=180)  # Adjust the image size and position as needed
        except Exception as e:
            print(f"Error adding image {i+1} to PDF: {str(e)}")
    
    # Save the PDF to a file
    pdf_file_path = 'lesson_plan_with_images.pdf'
    pdf.output(pdf_file_path)
    
    return pdf_file_path

# Review page function, now with multiple image generation
def review_page():
    st.markdown("<h3 class='main-header'>Review Cleaned Text</h3>", unsafe_allow_html=True)
    edited_text = st.text_area("Edit the cleaned text if necessary: ", value=st.session_state.cleaned_text, height=300)
    st.session_state.edited_text = edited_text

    # Input field to take the number of images from the user
    num_images = st.number_input("How many images would you like to generate?", min_value=1, max_value=10, step=1, value=1)

    # Back Button to go back to the Upload PDF page
    back_clicked = st.button("Back")
    send_clicked = st.button("Generate Lesson Plan and Illustration")
    
    if back_clicked:  # Go back to the upload page
        st.session_state.page = "upload"

    if send_clicked:  # Finalize the process and generate the lesson plan and images
        st.session_state.cleaned_text = st.session_state.edited_text
        st.info("Generating the lesson plan... Please wait!!!")
        lesson_plan = generate_lesson_plan(st.session_state.cleaned_text)
        summary = get_summary_text(lesson_plan, num_images)
        st.markdown(f"### Lesson Plan:\n\n{lesson_plan}")
        
        # Assume summary returns a dictionary {'image_1': 'description1', 'image_2': 'description2', ...}
        image_descriptions = summary
        
        # Generate prompts for each image description
        prompts = create_image_prompts(image_descriptions)
        
        # Generate images based on the created prompts
        st.info("Generating illustrations for the lesson!!!")

        image_urls = []
        with st.spinner('Generating images...'):
            for i, prompt in enumerate(prompts):
                # st.info(f"Generating image {i+1} of {num_images}")
                image_urls.extend(generate_images([prompt]))  # Generate one image per prompt
        
        if isinstance(image_urls, str) and image_urls.startswith("Error"):
            st.error(image_urls)
        else:
            # for i, image_url in enumerate(image_urls):
            #     st.image(image_url, caption=f"Lesson Illustration Image {i+1}")
            # Display each image with the description as the caption
            for (image_key, description), image_url in zip(image_descriptions.items(), image_urls):
                st.image(image_url, caption=f"{description}")
            
            # Generate PDF with lesson plan and images
            pdf_path = generate_pdf(lesson_plan, image_urls)
            with open(pdf_path, 'rb') as pdf_file:
                st.download_button(label="Download Lesson Plan with Illustrations", data=pdf_file, file_name="lesson_plan_with_images.pdf", mime="application/pdf")

# Validate user credentials
def validate_login(user_id, password, users_df):
    user_data = users_df[users_df['Id'] == user_id]
    if not user_data.empty and user_data.iloc[0]['Password'] == password:
        return user_data.iloc[0]['Name']
    return None

# Login page function
def login_page():
    st.markdown("<h1 class='main-header'>Lesson Plan Generator - Login</h1>", unsafe_allow_html=True)

    user_id = st.text_input("User ID")
    password = st.text_input("Password", type="password")

    login_clicked = st.button("Login")
    
    if login_clicked:  # Use this variable to check button click
        name = validate_login(user_id, password, users_df)
        if name:
            st.session_state.logged_in = True
            st.session_state.name = name
            # Set next page, then Streamlit will rerun the script
            st.session_state.page = "upload"  
            st.success(f"Logged in as {name}")

# Upload PDF page function
def upload_page():
    st.markdown(f"<p class='welcome'>Welcome to Jaipuria Institute of Management</p>", unsafe_allow_html=True)

    # Step 1: File uploader for the PDF
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    # Clear state only if a new file is uploaded
    if uploaded_file is not None:
        if st.session_state.uploaded_file != uploaded_file:
            st.session_state.pdf_text = ""
            st.session_state.cleaned_text = ""
            st.session_state.uploaded_file = uploaded_file  # Update uploaded file state

        # Step 2: Extract text from the uploaded PDF
        if st.session_state.pdf_text == "":
            pdf_text = extract_text_from_pdf(uploaded_file)
            st.session_state.pdf_text = pdf_text

        # Step 3: Send the extracted text to Groq for cleaning (only once)
        if st.session_state.cleaned_text == "":
            st.info("Cleaning the text for you. Please wait!!!")
            cleaned_text = get_clean_groqResponse(st.session_state.pdf_text)
            st.session_state.cleaned_text = cleaned_text
            st.success("Text cleaned successfully!!!")

    # Show the "Go to Review" button only after the text is cleaned
    if st.session_state.cleaned_text:
        go_to_review = st.button("Go to Review")
        if go_to_review:  # Button clicked, go to the review page
            st.session_state.page = "review"

# Initialize user session state
def initialize_session():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False  # Login the user
        st.session_state.name = ""
    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = ""  # Store extracted text from PDF
    if 'cleaned_text' not in st.session_state:
        st.session_state.cleaned_text = ""  # Store LLM cleaned text
    if 'edited_text' not in st.session_state:
        st.session_state.edited_text = ""  # Store edited text
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None  # Track uploaded file
    if 'page' not in st.session_state:
        st.session_state.page = "login"  # Set initial page to login

# Load the CSV file
def load_user_data():
    return pd.read_csv('user_info.csv')

# Load users data
users_df = load_user_data()

# CSS for custom styling (colors, fonts, and background)
st.markdown(
    """
    <style>
    /* Set background color for the main content area */
    .stApp {
        background-color: #ffe5b4;  /* Peach background */
    }
    .main-header {
        text-align: center;
        color: #cd7f32;
        font-family: 'Arial', sans-serif;
    }
    .sub-header {
        color: #ff6f61;
        font-size: 18px;
    }
    .welcome {
        font-size: 28px;
        font-weight: bold;
        color: #1a535c;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the logo at the top
st.image("Jaipuria logo.png", width=200)

# Initialize session state
initialize_session()

# Page handling
if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "upload":
    upload_page()
elif st.session_state.page == "review":
    review_page()