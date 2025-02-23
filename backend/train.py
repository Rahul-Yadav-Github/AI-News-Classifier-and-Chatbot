import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

# Define paths for saving models
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Train News Classifier Model
def train_news_classifier():
    print("Training News Classifier...")
    # Load the dataset
    DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'news_headlines_large.json')
    data = pd.read_json(DATA_FILE)

    # Preprocess data
    data['label'] = data['category'].map({'clickbait': 0, 'misinformation': 1, 'neutral': 2})
    X = data['headline']
    y = data['label']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the text
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # Save the model and vectorizer
    joblib.dump(model, os.path.join(MODEL_DIR, 'news_classifier_model.pkl'))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'news_classifier_vectorizer.pkl'))
    print("News Classifier Model and Vectorizer saved.")

# Train FAQ Chatbot Model
def train_faq_chatbot():
    print("Training FAQ Chatbot...")
    # Load the FAQ dataset
    DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'faq_data_large.json')
    with open(DATA_FILE, 'r') as f:
        faq_data = json.load(f)

    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode all FAQ questions
    faq_embeddings = model.encode([item['question'] for item in faq_data])

    # Save the embeddings and model locally
    joblib.dump(faq_embeddings, os.path.join(MODEL_DIR, 'faq_embeddings.pkl'))
    model.save(os.path.join(MODEL_DIR, 'faq_model'))
    print("FAQ Chatbot Model and Embeddings saved.")

if __name__ == '__main__':
    train_news_classifier()
    train_faq_chatbot()
