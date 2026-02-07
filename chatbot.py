import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

# Load sentences
with open('intents.txt', 'r') as file:
    sentences = file.read().splitlines()

# Predefined responses
responses = {
    "hello": "Hello! How can I help you?",
    "hi": "Hi there!",
    "hey": "Hey!",
    "how are you": "I'm just a bot, but I'm doing great!",
    "what is your name": "I am an NLP Chatbot created using Python.",
    "who created you": "I was created by Utsav during his internship task.",
    "what can you do": "I can answer your basic queries using NLP.",
    "what is python": "Python is a high-level, easy-to-learn programming language.",
    "what is ai": "AI stands for Artificial Intelligence, enabling machines to think.",
    "what is machine learning": "Machine Learning allows systems to learn from data.",
    "thanks": "You're welcome!",
    "bye": "Goodbye! Have a nice day!"
}

def preprocess(text):
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    return text

def get_response(user_input):
    user_input = preprocess(user_input)

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(sentences + [user_input])

    similarity = cosine_similarity(tfidf[-1], tfidf[:-1])
    index = similarity.argmax()

    matched = sentences[index]
    return responses.get(matched, "Sorry, I didn't understand that.")

print("AI Chatbot is running! Type 'bye' to exit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "bye":
        print("Bot:", responses["bye"])
        break

    print("Bot:", get_response(user_input))
