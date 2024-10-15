# Lesson Plan Generator with PDF Extraction and Illustration

This project is a Lesson Plan Generator that allows users to upload PDF documents, extract and clean the text, and generate structured lesson plans along with illustrative images. The generated lesson plans are downloadable as PDFs containing both the textual content and the generated illustrations. The application is built using Streamlit for the web interface, OpenAI's GPT-4 for text processing, and DALL-E for image generation. 

# Features 
PDF Text Extraction: The app extracts text from uploaded PDF files using pdfminer. 
Text Cleaning: The extracted text is cleaned and formatted using the GPT-4 model, making it more readable and suitable for lesson plans. 
Lesson Plan Generation: A structured lesson plan is created from the cleaned text, including sections such as introduction, main body, and class activities. 
Illustration Generation: Using DALL-E, the app generates educational illustrations based on the lesson plan content.  
PDF Generation: The lesson plan and the generated illustrations are compiled into a PDF, which can be downloaded. 
User Authentication: The app supports basic login functionality, using user credentials stored in a CSV file. 

# Dependencies 
To run this project, ensure you have the following dependencies installed: 

fpdf – for generating PDFs 
openai – for interacting with OpenAI's API (GPT-4 and DALL-E) 
pdfminer – for extracting text from PDFs 
requests – for handling image downloads  
streamlit – for creating the web interface 
dotenv – for loading environment variables  
pandas – for reading user data from a CSV file 

# Usage 

## Setting Up Environment Variables  
You need to set up your environment variables to store the API keys required for interacting with OpenAI. Create a .env file in the project directory and add the following keys: 

api_key=your_groq_api_key 
openai_api_key=your_openai_api_key 
Make sure to replace your_groq_api_key and your_openai_api_key with your actual API keys. 

## Running the Application 
To run the Streamlit app, execute the following command: 
streamlit run app.py 

## Workflow 
Login Page: Users must log in using their credentials, which are stored in a user_info.csv file. 
PDF Upload: Once logged in, the user can upload a PDF file. The text is extracted, cleaned, and prepared for lesson generation. 
Review and Generate: Users can review and edit the cleaned text before generating the lesson plan and choosing the number of illustrations to generate. 
Download: After generating the lesson plan and illustrations, users can download a PDF that includes both the text and images. 

## Customization
Lesson Plan Structure: The structure of the generated lesson plan is customizable by modifying the generate_lesson_plan function.  
Illustrations: The prompts for image generation can be fine-tuned in the create_image_prompts function. 
File Structure 

├── .env                         # Environment variables  
├── app.py                       # Main Streamlit application code 
├── user_info.csv                # CSV containing user credentials (ID, Password, Name) 
├── Jaipuria logo.png            # Institute's logo displayed in the app 
└── README.md                    # This file 

# Key Functions 
extract_text_from_pdf(): Extracts text from an uploaded PDF file. 
get_clean_groqResponse(): Sends the extracted text to GPT-4 (or Groq) for cleaning and formatting. 
generate_lesson_plan(): Generates a lesson plan based on the cleaned text using GPT-4. 
create_image_prompts(): Creates image prompts based on the lesson plan content. 
generate_images(): Uses OpenAI's DALL-E to generate illustrations. 
generate_pdf(): Compiles the lesson plan and images into a downloadable PDF. 
