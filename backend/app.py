import os
from flask import Flask, request, jsonify
from models.news_classifier import classify_headline
from models.faq_chatbot import chatbot_response

app = Flask(__name__)

@app.route('/classify_headline', methods=['POST'])
def api_classify_headline():
    headline = request.json['headline']
    prediction, score = classify_headline(headline)
    return jsonify({'prediction': prediction, 'score': score})

@app.route('/chatbot', methods=['POST'])
def api_chatbot():
    query = request.json['query']
    response, confidence = chatbot_response(query)
    return jsonify({'response': response, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)