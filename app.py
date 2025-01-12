import streamlit as st
import pickle
import numpy as np
import re
import emoji
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation

## Slang dictionary
slang_dic = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    'ur' : 'you are',
    'lol' : 'laughing out loud',
    'wth' : 'what the hell',    
    "mightn't": "might not",
}


# Load the trained models
with open('svm_w2v.pkl', 'rb') as file:
    svm_w2v_model = pickle.load(file)

with open('svm_glove.pkl', 'rb') as file:
    svm_glove_model = pickle.load(file)

with open('svm_tfidf.pkl', 'rb') as file:
    svm_tfidf_model = pickle.load(file)

with open('svm_count.pkl', 'rb') as file:
    svm_count_model = pickle.load(file)

with open('nb_onehot.pkl', 'rb') as file:
    nb_model = pickle.load(file)

with open('svm_w2v_skipgram.pkl', 'rb') as file:
    svm_w2v_skipgram = pickle.load(file)

# Load Word2Vec model
with open('word2vec_model.pkl', 'rb') as file:
    word2vec_model = pickle.load(file)

# Load GloVe embeddings
glove_embeddings = {}
with open("glove.6B.300d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        glove_embeddings[word] = vector

# Initialize tokenizer, stopwords, and lemmatizer
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
stop_words_set = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Pre-processing function
def removal_html(text):
    return re.sub(r'<.*?>', '', text)

def removal_url(text):
    clean_text = re.sub(r'https?:\/\/[^\s]+', '', text)
    return re.sub(r'www\.[a-z]?\.?(com)+|[a-z]*?\.?(com)+', '', clean_text)

def twitter_handles(text):
    return re.sub(r'@\w*', '', text)

def pre_processing(text):
    text = emoji.demojize(text)
    text = ' '.join([slang_dic.get(word.lower(), word) for word in text.split()])
    text = twitter_handles(text)
    text = removal_url(text)
    text = removal_html(text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\d+', '', text)
    text = ''.join([char if char not in punctuation else ' ' for char in text])
    text = ' '.join([word for word in text.lower().split() if word not in stop_words_set])
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def post_processing(text):
    tokens = tokenizer.tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    text = ' '.join(tokens).lower()
    return text

# Function to convert a preprocessed tweet into Word2Vec embeddings
def get_word2vec_embedding(tweet, model, vector_size=300):
    tokens = tweet.split()
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)

# Function to convert a preprocessed tweet into GloVe embeddings
def get_glove_embedding(tweet, glove_embeddings, vector_size=300):
    tokens = tweet.split()
    vectors = [glove_embeddings[token] for token in tokens if token in glove_embeddings]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)

# Function to predict sentiment using a model
def predict_sentiment(model, input_data):
    if isinstance(input_data, np.ndarray):
        input_data = input_data.reshape(1, -1)
    else:
        input_data = [input_data]  
    prediction = model.predict(input_data)
    return prediction

# Function to beautify sentiment output
def beautify_output(sentiment):
    if sentiment == 'Positive':
        return "Positive ⬆️ 💚"  # Green heart for positive sentiment
    elif sentiment == 'Neutral':
        return "Neutral ➡️ 💛"   # Yellow heart for neutral sentiment
    else:
        return "Negative ⬇️ 💔"  # Broken heart for negative sentiment

# Streamlit App UI
st.title("Tweet Sentiment Analysis")
st.write("This app allows you to analyze the sentiment of a tweet using multiple models.")

# Input: Enter a tweet
tweet_input = st.text_area("Enter your tweet:")

# If the user submits the tweet
# If the user submits the tweet
if st.button("Analyze Sentiment"):
    if tweet_input:
        # Pre-process the user input
        tweet_processed = pre_processing(tweet_input)
        tweet_post_processed = post_processing(tweet_processed)

        # Get Word2Vec and GloVe embeddings
        word2vec_embedding = get_word2vec_embedding(tweet_post_processed, word2vec_model)
        glove_embedding = get_glove_embedding(tweet_post_processed, glove_embeddings)

        # Predict for all models
        prediction_svm_w2v = predict_sentiment(svm_w2v_model, word2vec_embedding)
        prediction_svm_glove = predict_sentiment(svm_glove_model, glove_embedding)
        prediction_svm_tfidf = predict_sentiment(svm_tfidf_model, tweet_post_processed)
        prediction_svm_count = predict_sentiment(svm_count_model, tweet_post_processed)
        prediction_nb = predict_sentiment(nb_model, tweet_post_processed)
        prediction_svm_skipgram = predict_sentiment(svm_w2v_skipgram, word2vec_embedding)

        # Create a bordered table with a "Models" and "Output" column
        st.markdown("""
        <style>
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 18px;
            text-align: left;
            table-layout: fixed;
        }
        th, td {
            padding: 12px;
            border: 1px solid #ddd;
            text-align: center;
        }
        th {
            background-color: #f4f4f4;
            color: #ff4b4b;  /* Change the header font color */
        }
        </style>
        """, unsafe_allow_html=True)

        # Start of the HTML Table structure
        st.markdown("""
        <table>
            <tr>
                <th>Models</th>
                <th>Output</th>
            </tr>
            <tr>
                <td>SVM with Word2Vec</td>
                <td>{}</td>
            </tr>
            <tr>
                <td>SVM with GloVe</td>
                <td>{}</td>
            </tr>
            <tr>
                <td>SVM with TF-IDF</td>
                <td>{}</td>
            </tr>
            <tr>
                <td>Naive Bayes (Unigram+Bigram)</td>
                <td>{}</td>
            </tr>
            <tr>
                <td>SVM with Count Vectors</td>
                <td>{}</td>
            </tr>
            <tr>
                <td>SVM with FastText</td>
                <td>{}</td>
            </tr>
        </table>
        """.format(
            beautify_output(prediction_svm_w2v[0]),
            beautify_output(prediction_svm_glove[0]),
            beautify_output(prediction_svm_tfidf[0]),
            beautify_output(prediction_nb[0]),
            beautify_output(prediction_svm_count[0]),
            beautify_output(prediction_svm_skipgram[0])
        ), unsafe_allow_html=True)
