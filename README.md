# CV Matcher and Motivation Letter Generation

## Project Overview

This project is a web application designed to assist users in matching resumes with job descriptions and generating tailored motivation letters. 
## Main Features

### Resume Matching

- Users can upload multiple resume files in various formats, including PDF, DOCX, TXT, and image files.
- The application extracts and preprocesses the text from the uploaded resumes.
- It compares the resumes against a job description provided by the user, calculating similarity scores using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization and cosine similarity metrics.
- The top matching resumes are displayed to the user for easy review.

### Motivation Letter Generation

- The application utilizes OpenAI's GPT-3.5 model to generate a motivation letter tailored to the job description provided by the user.
- This letter serves as a helpful template for users to customize further, enhancing their application package.

### File Handling

- The application supports various file types, enabling flexibility for users to upload resumes in different formats without the need for extensive conversions.

## Technologies Used

- **Flask**: A lightweight web framework for Python, used to create the web application. It handles routing, rendering HTML templates, and processing form submissions.
- **HTML/CSS**: Used for the front-end design of the application, providing an intuitive user interface for uploading resumes and entering job descriptions.

### Python Libraries

- **docx2txt**: For extracting text from DOCX files.
- **PyPDF2**: For extracting text from PDF files.
- **pytesseract**: For performing OCR (Optical Character Recognition) to extract text from images.
- **PIL (Pillow)**: For image processing tasks.
- **nltk**: For natural language processing tasks, including stopword removal and lemmatization.
- **scikit-learn**: For vectorizing text and calculating cosine similarity to compare resumes with the job description.
- **OpenAI**: For generating motivation letters using the OpenAI API.

### Tesseract OCR:

- An open-source OCR engine used to extract text from image files. The application specifies the path to the Tesseract executable to facilitate this.
