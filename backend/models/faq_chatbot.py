import json
import joblib
from sentence_transformers import SentenceTransformer, util
import torch

def train_model():
    with open('../../data/faq_data_large.json', 'r') as f:
        faq_data = json.load(f)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    faq_embeddings = model.encode([item['question'] for item in faq_data])

    joblib.dump(faq_embeddings, 'faq_embeddings.pkl')
    model.save('faq_model')

model = SentenceTransformer('faq_model')
faq_embeddings = joblib.load('faq_embeddings.pkl')

with open('../../data/faq_data_large.json', 'r') as f:
    faq_data = json.load(f)

def chatbot_response(query):
    query_embedding = model.encode([query])
    cos_scores = util.pytorch_cos_sim(query_embedding, faq_embeddings)[0]
    best_match_index = torch.argmax(cos_scores).item()
    best_match_score = cos_scores[best_match_index].item()
    
    if best_match_score >= 0.5:
        return faq_data[best_match_index]['answer'], best_match_score
    else:
        return "I'm not sure about this. Let me find more details.", best_match_score

if __name__ == '__main__':
    train_model()
