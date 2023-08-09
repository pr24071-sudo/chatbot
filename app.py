import csv
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from flask import Flask, request, render_template, jsonify, session
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

qa_pairs = {}
with open('combined.csv', 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        question = row['question']
        answer = row['answer']
        qa_pairs[question] = answer

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

questions = list(qa_pairs.keys())
tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words=stop_words)
tfidf_matrix = tfidf_vectorizer.fit_transform(questions)
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)


def get_most_similar_question(user_input):
    input_vector = tfidf_vectorizer.transform([user_input])
    similarities = cosine_similarity(input_vector, tfidf_matrix)[0]
    most_similar_idx = np.argmax(similarities)
    return questions[most_similar_idx]


def contains_name(input_text, name):
    return name.lower() in input_text.lower()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get_response', methods=['POST'])
def get_response():
    response="Hello there! Welcome to the chatbot. Please provide your role (e.g., Doctor, Nurse, Health Care Assistant)."
    user_input = request.form['user_input']

    if "role" not in session:
        session["role"] = user_input
        response = "Please provide your department(e.g., Inpatient, Outpatient, Maternity or Emergency.)"
        

    elif "department" not in session:
        session["department"] = user_input
        response = "Thank you for specifying your department. How can I assist you today?"
    else:
        if user_input.lower() == 'bye' or user_input.lower() == 'exit' or user_input.lower() == 'quit':
            response = "Bye! See you soon!"
        if user_input.lower() =='hi' or user_input.lower() =='hello' or user_input.lower()=='hey' or user_input.lower()=='greetings':
            response='Hello! How may I help you?'
        elif user_input in qa_pairs:
            response = qa_pairs[user_input]
        else:
            most_similar_question = get_most_similar_question(user_input)
            response = qa_pairs[most_similar_question]
            if most_similar_question.lower() != user_input.lower():
                response += f" (Did you mean: '{most_similar_question}')"
    
    return jsonify({'response': response})


if __name__ == '__main__':
    app.secret_key = 'ABCD1234'
    app.run(debug=True)
