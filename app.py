import pandas as pd
import numpy as np
from flask import Flask, render_template_string, request, jsonify
from pyngrok import ngrok
import threading
from datasets import load_dataset
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask_cors import CORS
import re
import os

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('vader_lexicon')

class DatasetCareerAdvisor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None

    def load_dataset(self):
        """Load and prepare the career guidance dataset"""
        print("Loading dataset...")
        try:
            dataset = load_dataset("mb7419/career-guidance-reddit")
            self.df = pd.DataFrame(dataset['train'])
            print(f"Loaded {len(self.df)} records")
            return self.df
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            # Create a sample dataset if loading fails
            self.df = pd.DataFrame({
                'question_content': [
                    "How do I become a software developer?",
                    "Should I switch to data science?",
                    "Is medical school worth it?"
                ],
                'body': [
                    "Learn programming languages and build projects.",
                    "Focus on statistics and machine learning.",
                    "Consider the long-term benefits and challenges."
                ]
            })
            print("Created sample dataset")
            return self.df

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)

    def categorize_fields(self):
        """Categorize posts into career fields"""
        print("Categorizing career fields...")
        field_keywords = {
            'CS/IT': ['computer science', 'programming', 'software', 'coding', 'developer', 'tech', 'IT',
                     'data science', 'web development', 'cybersecurity', 'devops', 'frontend', 'backend'],
            'Medical': ['medicine', 'doctor', 'healthcare', 'hospital', 'medical', 'physician', 'nurse',
                       'dental', 'pharmacy', 'clinical', 'health care'],
            'Engineering': ['engineer', 'engineering', 'mechanical', 'electrical', 'civil', 'aerospace',
                          'chemical', 'industrial'],
            'Business': ['business', 'finance', 'management', 'marketing', 'accounting', 'consulting',
                        'sales', 'entrepreneur', 'MBA'],
            'Creative': ['design', 'art', 'music', 'writing', 'media', 'film', 'animation', 'graphic',
                        'content creation'],
            'Education': ['teacher', 'professor', 'education', 'teaching', 'academic', 'school',
                         'instructor', 'tutor'],
            'Legal': ['law', 'legal', 'attorney', 'lawyer', 'paralegal', 'compliance', 'regulatory'],
            'Trade': ['electrician', 'plumber', 'carpenter', 'mechanic', 'construction', 'technician',
                     'manufacturing']
        }

        self.df['fields'] = 'Other'
        for field, keywords in field_keywords.items():
            pattern = '|'.join(r'(?:\b{}\b)'.format(k) for k in keywords)
            mask = self.df['question_content'].str.lower().str.contains(pattern, regex=True, na=False)
            self.df.loc[mask, 'fields'] = field

    def extract_career_stage(self):
        """Detect career stage from question content"""
        print("Extracting career stages...")
        stage_patterns = {
            'Student': r'\b(student|university|college|school|studying|degree|freshman|sophomore|junior|senior)\b',
            'Entry Level': r'\b(graduate|entry level|junior|starting|new grad|first job|internship)\b',
            'Mid Career': r'\b(experienced|mid level|senior|5\+? years|professional)\b',
            'Career Change': r'\b(change career|switch|transition|pivot|switching|changing)\b'
        }

        self.df['career_stage'] = 'Unknown'
        for stage, pattern in stage_patterns.items():
            mask = self.df['question_content'].str.lower().str.contains(pattern, regex=True, na=False)
            self.df.loc[mask, 'career_stage'] = stage

    def analyze_sentiment_and_intention(self):
        """Analyze sentiment and classify post intentions"""
        print("Analyzing sentiment and intentions...")
        self.df['question_sentiment'] = self.df['question_content'].apply(
            lambda x: self.sia.polarity_scores(str(x))['compound'] if pd.notna(x) else 0
        )
        self.df['response_sentiment'] = self.df['body'].apply(
            lambda x: self.sia.polarity_scores(str(x))['compound'] if pd.notna(x) else 0
        )
        
        # Classifying sentiment as Positive or Negative
        self.df['response_sentiment_class'] = self.df['response_sentiment'].apply(
            lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral')
        )

    def prepare_model(self):
        """Prepare the TF-IDF model"""
        print("Preparing TF-IDF model...")
        try:
            processed_questions = self.df['question_content'].apply(self.preprocess_text)
            self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), 
                                            min_df=2, max_df=0.90, max_features=10000)
            self.tfidf_matrix = self.vectorizer.fit_transform(processed_questions)
            print("Model preparation complete")
        except Exception as e:
            print(f"Error in model preparation: {str(e)}")
            raise

    def get_advice(self, query, top_n=3, min_similarity=0.15):
        """Get career advice based on query"""
        try:
            processed_query = self.preprocess_text(query)
            query_vector = self.vectorizer.transform([processed_query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            top_indices = similarities.argsort()[-top_n:][::-1]

            results = {'positive': [], 'negative': [], 'neutral': []}
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity > min_similarity:
                    result = self.df.iloc[idx]
                    sentiment = result['response_sentiment_class']
                    response_data = {
                        'similarity': round(similarity * 100, 2),
                        'field': result['fields'],
                        'career_stage': result['career_stage'],
                        'question': result['question_content'],
                        'response': result['body'],
                        'sentiment': sentiment
                    }
                    results[sentiment.lower()].append(response_data)
            
            return results
        except Exception as e:
            print(f"Error in getting advice: {str(e)}")
            return {'positive': [], 'negative': [], 'neutral': []}

# Initialize Flask app and advisor
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})



# Initialize advisor and load dataset
advisor = DatasetCareerAdvisor()
advisor.load_dataset()
advisor.categorize_fields()
advisor.extract_career_stage()
advisor.analyze_sentiment_and_intention()
advisor.prepare_model()

@app.route("/", methods=["POST"])
def home():
    if request.method == "POST":
        data = request.form  # Get form data
        query = data.get("query", "").strip()  # Extract the query safely
        
        if not query:
            return jsonify({"error": "Query is required"}), 400  # Handle empty queries

        advice = advisor.get_advice(query)  # Get advice from the advisor module
        
        return jsonify({"advice": advice})  # Return JSON response

def run_flask():
    app.run(host="0.0.0.0", port=8000)


# Start Flask server with threading
if __name__ == "__main__":
    threading.Thread(target=run_flask).start()



