from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask import Flask, render_template, request
import pdfplumber
import spacy
from sentence_transformers import SentenceTransformer, util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Flask app
app = Flask(__name__)

# Load fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('/Users/zainnofal/Projects/Resume Catergorizer/resume_category_model')
tokenizer = AutoTokenizer.from_pretrained('/Users/zainnofal/Projects/Resume Catergorizer/resume_category_model')

# Categories from your model
categories = [
    "Accountant", "Advocate", "Agriculture", "Apparell", "Arts", 
    "Automobile", "Aviation", "Banking", "BPO", "Business Development",
    "Chef", "Construction", "Consultant", "Designer", "Digital Media",
    "Engineering", "Finance", "Fitness", "Healthcare", "HR",
    "Information Technology", "Public Relations", "Sales", "Teacher"
]

# Initialize SpaCy model for feature extraction
nlp = spacy.load("en_core_web_sm")

# Predefined list of skills and education keywords
skills_list = ["accounting", "finance", "auditing", "management", "data analysis", "excel", "python", "teamwork", "communication"]
education_keywords = ["bachelor", "master", "phd", "degree", "university", "college", "diploma", "MBA", "MS", "BA"]

def preprocess_text(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    return inputs

def predict(text):
    inputs = preprocess_text(text)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions.item()

def get_category_name(label_id):
    return categories[label_id]

# Function to extract resume features (skills, education, and experience)
def extract_features(resume_text):
    doc = nlp(resume_text)
    features = {
        "education": [],
        "skills": [],
        "experience": [],
    }

    # Extract education-related entities based on specific keywords
    for ent in doc.ents:
        if any(keyword.lower() in ent.text.lower() for keyword in education_keywords):
            features["education"].append(ent.text)

    # Extract skills based on the predefined list
    for token in doc:
        if token.text.lower() in skills_list:
            features["skills"].append(token.text)

    # Extract experience-related information using regex for common phrases
    experience_keywords = ["worked", "experience", "responsible for", "managed", "led", "performed"]
    for sent in doc.sents:
        for keyword in experience_keywords:
            if keyword in sent.text.lower():
                features["experience"].append(sent.text)

    # Remove duplicates in skills and experience
    features["skills"] = list(set(features["skills"]))
    features["experience"] = list(set(features["experience"]))
    
    return features

def compute_similarity(resume_text, job_description):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_embedding = model.encode(job_description, convert_to_tensor=True)
    return util.pytorch_cos_sim(resume_embedding, job_embedding).item()

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment["compound"]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the job description and uploaded resume
        job_description = request.form.get("job_description")
        file = request.files["resume"]

        # Extract text from the resume PDF
        resume_text = ""
        if file:
            with pdfplumber.open(file) as pdf:
                resume_text = " ".join(page.extract_text() for page in pdf.pages)

        # Extract features from resume
        features = extract_features(resume_text)

        # Compute similarity score
        similarity = compute_similarity(resume_text, job_description)

        # Analyze sentiment
        sentiment_score = analyze_sentiment(resume_text)

        # Calculate overall score
        skill_score = len(features["skills"])
        experience_score = len(features["experience"])
        overall_score = 0.4 * similarity + 0.3 * skill_score + 0.2 * experience_score + 0.1 * sentiment_score

        # Predict the category using the fine-tuned model
        predicted_label_id = predict(resume_text)
        predicted_category = get_category_name(predicted_label_id)

        return render_template("result.html", 
                               similarity=similarity, 
                               sentiment_score=sentiment_score, 
                               overall_score=overall_score,
                               best_field=predicted_category,  # Use predicted_category here
                               skills=features["skills"], 
                               experience=features["experience"],
                               education=features["education"], 
                               predicted_category=predicted_category)  # Pass predicted category to the template

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
