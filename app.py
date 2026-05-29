import os
import logging
from flask import Flask, render_template, request
import PyPDF2
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Clear Warnings ---
os.environ["HF_HUB_OFFLINE"] = "1"
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- 2. Initialize App ---
app = Flask(__name__)

# --- 3. Load AI Model ---
print("Loading AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model Loaded!")

# --- 4. Helper Functions ---
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() + " "
    except Exception as e:
        print(f"PDF Error: {e}")
        return None
    return text

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def calculate_match_score(resume_text, job_description):
    embeddings = model.encode([resume_text, job_description])
    resume_vector = embeddings[0].reshape(1, -1)
    jd_vector = embeddings[1].reshape(1, -1)
    score = cosine_similarity(resume_vector, jd_vector)[0][0]
    return round(score * 100, 2)

# --- 5. The Web Route (This prevents the 404 error!) ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get data from the HTML form
        job_desc = request.form.get('job_desc')
        resume_file = request.files.get('resume')

        if not job_desc or not resume_file or resume_file.filename == '':
            return render_template('index.html', error="Please provide both inputs.")

        # Process the PDF
        raw_resume = extract_text_from_pdf(resume_file)
        if not raw_resume:
            return render_template('index.html', error="Could not read the PDF.", job_desc=job_desc)

        # Clean and Score
        clean_resume = clean_text(raw_resume)
        clean_jd = clean_text(job_desc)
        score = calculate_match_score(clean_resume, clean_jd)

        # Determine Verdict
        if score > 70:
            verdict = "Strong Candidate! Move to interview phase."
        elif score > 40:
            verdict = "Potential Match. Review manually."
        else:
            verdict = "Weak Match. Lacks core semantic alignment."

        # Send data back to the HTML template
        return render_template('index.html', score=score, verdict=verdict, job_desc=job_desc)

    # If just visiting the page, load the empty form
    return render_template('index.html', score=None)

# --- 6. Run Server ---
if __name__ == '__main__':
    print("🚀 Website is running at: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)