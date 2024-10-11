from flask import Flask, request, render_template
import os
import docx2txt
import PyPDF2
import pytesseract
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import openai  


pytesseract.pytesseract.tesseract_cmd = r'C:\Users\dynabook\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

lemmatizer = WordNetLemmatizer()

# Function to extract text from PDF files
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to extract text from DOCX files
def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

# Function to extract text from TXT files
def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to extract text from image files using Tesseract OCR
def extract_text_from_image(file_path):
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img)
    return text


def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    elif file_path.endswith(('.jpg', '.jpeg', '.png')):
        return extract_text_from_image(file_path)
    else:
        return ""

# Function to clean and preprocess text
def preprocess_text(text):
   
    text = text.lower()  
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) 
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)

    # Lemmatize words
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())

    return text


def generate_motivation_letter(job_description):
    openai.api_key = "# Add your OpenAI API key here"  

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Please generate a motivation letter for the following job description: {job_description}"
            }
        ]
    )

    return response.choices[0].message['content'].strip()

@app.route("/")
def matchresume():
    return render_template('matchresume.html')

@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resumes')

        # Preprocess job description
        job_description = preprocess_text(job_description)

        resumes = []
        for resume_file in resume_files:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(filename)
            extracted_text = extract_text(filename)
            resumes.append(preprocess_text(extracted_text)) 

        if not resumes or not job_description:
            return render_template('matchresume.html', message="Please upload resumes or images and enter a job description.")

        # Vectorize job description and resumes
        vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
        vectors = vectorizer.toarray()

        job_vector = vectors[0]
        resume_vectors = vectors[1:]
        similarities = cosine_similarity([job_vector], resume_vectors)[0]

        top_indices = similarities.argsort()[-5:][::-1]
        top_resumes = [resume_files[i].filename for i in top_indices]
        similarity_scores = [round(similarities[i], 2) for i in top_indices]

        # Generate motivation letter
        motivation_letter = generate_motivation_letter(job_description)

        return render_template('matchresume.html',
                               message="Top matching resumes/images:",
                               top_resumes=top_resumes,
                               similarity_scores=similarity_scores,
                               motivation_letter=motivation_letter)  

    return render_template('matchresume.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
