import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model():
    data = pd.read_json('../../data/news_headlines_large.json')
    data['label'] = data['category'].map({'clickbait': 0, 'misinformation': 1, 'neutral': 2})
    X = data['headline']
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_tfidf, y_train)

    joblib.dump(model, 'news_classifier_model.pkl')
    joblib.dump(vectorizer, 'news_classifier_vectorizer.pkl')

model = joblib.load('news_classifier_model.pkl')
vectorizer = joblib.load('news_classifier_vectorizer.pkl')

def classify_headline(headline):
    headline_tfidf = vectorizer.transform([headline])
    prediction = model.predict(headline_tfidf)[0]
    probabilities = model.predict_proba(headline_tfidf)[0]
    conspiracy_score = probabilities[1] * 100  # Assuming index 1 is for misinformation
    return prediction, conspiracy_score

if __name__ == '__main__':
    train_model()
