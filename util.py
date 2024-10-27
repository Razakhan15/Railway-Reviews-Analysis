import pickle
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import re

with open('sentiment_analysis_model.pkl', 'rb') as file:
    loaded_pipeline = pickle.load(file)

with open('tokenizer.pkl', 'rb') as file:
    loaded_tokenizer = pickle.load(file)

loaded_model = AutoModelForSequenceClassification.from_pretrained('roberta_model')

def polarity_scores_roberta(text):
    encoded_text = loaded_tokenizer(text, return_tensors='pt')
    output = loaded_model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    max_index = scores.argmax()
    return {
        'roberta_neg': max_index == 0,
        'roberta_neu': max_index == 1,
        'roberta_pos': max_index == 2,
    }

def clean_tweet(tweet):
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'http\S+|www\S+', '', tweet)
    tweet = re.sub(r'#\w+', '', tweet)
    tweet = tweet.strip()
    return tweet

def get_review_analysis(text):
    cleaned_text = clean_tweet(text)
    label_mapping = {
    0: 'Cleanliness',
    1: 'Others',
    2: 'Medical issues',
    3: 'Food Services',
    4: 'Train Delay',
    5: 'Ticket issues',
    }
    category_pred = loaded_pipeline.predict([cleaned_text])[0]
    category = label_mapping.get(category_pred, 'Unknown')

    roberta_booleans = polarity_scores_roberta(cleaned_text)
    
    return {
        'category': category,
        **roberta_booleans
    }

example_text = "The train was delayed for 2 hours."
result = get_review_analysis(example_text)
print(result)
