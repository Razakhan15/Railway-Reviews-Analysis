from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
from flaskext.mysql import MySQL
import pickle
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import re
from pymysql.cursors import DictCursor

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
mysql = MySQL(cursorclass=DictCursor)
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'sentiment_analysis'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)
conn = mysql.connect()
cursor = conn.cursor()

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

@app.route('/fetch-data', methods=['GET'])
@cross_origin()
def fetch():
    cursor.execute('SELECT * FROM `reviews`')
    res = cursor.fetchall()
    return res

@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def upload():
    text = request.get_json()['review']
    email = request.get_json()['email']
    cleaned_text = clean_tweet(text)
    label_mapping = {
    0: 'cleanliness',
    1: 'others',
    2: 'medical_issues',
    3: 'food_services',
    4: 'train_delay',
    5: 'ticket_issues',
    }
    category_pred = loaded_pipeline.predict([cleaned_text])[0]
    category = label_mapping.get(category_pred, 'Unknown')
    roberta_booleans = polarity_scores_roberta(cleaned_text)
    cursor.execute("INSERT INTO reviews (`id`, `review`, `email`, `positive`, `negative`, `neutral`, `category`) VALUES (NULL, %s, % s, % s, % s, % s, % s)", (cleaned_text,email,roberta_booleans['roberta_pos'],roberta_booleans['roberta_neg'],roberta_booleans['roberta_neu'], category))
    conn.commit()
    return "success"

if __name__ == '__main__':
    app.run()