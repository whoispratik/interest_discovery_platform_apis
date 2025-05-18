
import uvicorn
from fastapi import FastAPI
from input_params import get_description,get_info
import cloudpickle
import re
import string
from fastapi.middleware.cors import CORSMiddleware
import pickle  
from collections import Counter
from typing import Dict
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


def load_subreddit_mapping() -> Dict[str, tuple]:
    with open("subreddit_mapping.pkl", "rb") as f:
        return pickle.load(f)


subreddit_mapping = load_subreddit_mapping()

@app.post('/')
def index():
    return {'message': 'Welcome to Interest based ML API'}

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str, max_length: int = 10000) -> str:
    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[\"']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-z\s]", '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    truncated_text = ' '.join(tokens[:max_length])

    return truncated_text
@app.post("/predict_category")
def predict_category(data: get_description):
    clf = joblib.load("multinomial_nb_model.pkl") 
    vectorizer = joblib.load("tfidf_vectorizer.pkl") 

    input_text = preprocess_text(f"{data.title} {data.description}",len(data.description.split()))
    X_input = vectorizer.transform([input_text]) 
    pred_label = clf.predict(X_input)[0]

    category_1, category_2 = subreddit_mapping.get(pred_label, ("Unknown", "Unknown"))

    return {
        "predicted_subreddit": pred_label,
        "category_1": category_1,
        "category_2": category_2
    }

@app.post("/interest_prediction") 
def interest_prediction(data: get_info):
    clf = joblib.load("multinomial_nb_model.pkl")  
    vectorizer = joblib.load("tfidf_vectorizer.pkl")  
    info = data.posts + data.likes + [
        {"title": comment["post"]["title"], "description": comment["post"]["description"], "comment": comment["comment"]}
        for comment in data.comments
    ]
    subreddit_counter = Counter()
    analyzer = SentimentIntensityAnalyzer()
    for item in info:
        if "comment" in item:
            
            sentiment_scores = analyzer.polarity_scores(item["comment"])
            compound_score = sentiment_scores['compound']

        
            if compound_score < -0.2:
                continue

        input_text = preprocess_text(f"{item['title']} {item['description']}")
        X_input = vectorizer.transform([input_text]) 
        pred_label = clf.predict(X_input)[0] 
        subreddit_counter[pred_label] += 1 

    max_count = max(subreddit_counter.values(), default=0)
    max_subreddits = [subreddit for subreddit, count in subreddit_counter.items() if count == max_count]

    max_category_to_subreddits = {}
    for subreddit in max_subreddits:
        category1, category2 = subreddit_mapping.get(subreddit, ('Unknown', 'Unknown'))

        if category1 not in max_category_to_subreddits:
            max_category_to_subreddits[category1] = []
        max_category_to_subreddits[category1].append(subreddit)

        if category2 not in max_category_to_subreddits:
            max_category_to_subreddits[category2] = []
        max_category_to_subreddits[category2].append(subreddit)

  
    for category, subreddits in max_category_to_subreddits.items():
        if len(subreddits) == 1:
            max_category_to_subreddits[category] = subreddits[0]

    return max_category_to_subreddits
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=9000)
