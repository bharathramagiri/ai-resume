import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + " "
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text
import re

def clean_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    # Convert to lowercase
    return text.lower().strip()
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained AI model (it will download the first time you run it)
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_match_score(resume_text, job_description):
    # 1. Convert both texts into AI embeddings
    embeddings = model.encode([resume_text, job_description])
    
    # embeddings[0] is the resume, embeddings[1] is the JD
    resume_vector = embeddings[0].reshape(1, -1)
    jd_vector = embeddings[1].reshape(1, -1)
    
    # 2. Calculate the cosine similarity between the two vectors
    # This returns a score between 0 (no match) and 1 (perfect match)
    score = cosine_similarity(resume_vector, jd_vector)[0][0]
    
    # 3. Convert to a percentage
    return round(score * 100, 2)
def main():
    # --- 1. Define your inputs ---
    # Make sure you have a sample resume PDF in the same folder, or update the path
    resume_path = 'sample_resume.pdf' 
    
    job_description = """
    We are looking for a Software Engineer with experience in Python, 
    machine learning, and natural language processing. The ideal candidate 
    should have experience building REST APIs and deploying models to production.
    """
    
    # --- 2. Run the pipeline ---
    print("Extracting text from resume...")
    raw_resume = extract_text_from_pdf(resume_path)
    
    if not raw_resume:
        print("Could not extract text. Please check the PDF.")
        return

    print("Cleaning data...")
    clean_resume = clean_text(raw_resume)
    clean_jd = clean_text(job_description)
    
    print("Calculating AI match score...")
    match_percentage = calculate_match_score(clean_resume, clean_jd)
    
    # --- 3. Output the result ---
    print("-" * 30)
    print(f"Resume Match Score: {match_percentage}%")
    print("-" * 30)

    if match_percentage > 70:
        print("Verdict: Strong Candidate! Move to interview phase.")
    elif match_percentage > 40:
        print("Verdict: Potential Match. Review manually.")
    else:
        print("Verdict: Weak Match.")

if __name__ == "__main__":
    main()