import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
import emoji

# Load pre-trained models
@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

svm_count = load_model('svm_count.pkl')
svm_tfidf = load_model('svm_tfidf.pkl')

# Preprocessing functions
slang_dic = {"ur": "you are", "lol": "laughing out loud", "wth": "what the hell"}  # Example slang dictionary
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
stop_words_set = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
punctuation = set(string.punctuation)

def removal_html(text):
    return re.sub(r'<.*?>', '', text)

def removal_url(text):
    clean_text = re.sub(r'https?://[^\s]+', '', text)
    return re.sub(r'www\.[a-z]?\.?[a-z]*\.?(com)+', '', clean_text)

def twitter_handles(text):
    return re.sub(r'@\w*', '', text)

def pre_processing(text):
    text = emoji.demojize(text)
    text = ' '.join([slang_dic.get(word, word) for word in text.split()])
    text = twitter_handles(text)
    text = removal_url(text)
    text = removal_html(text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\d+', '', text)
    text = ''.join([char if char not in punctuation else ' ' for char in text])
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words_set])
    text = re.sub(r'\s+', ' ', text)
    return text

# Streamlit app
st.title("Sentiment Analysis App")
st.write("Upload your dataset or input text for sentiment analysis.")

# Upload and preprocess text data
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data:")
    st.write(df.head())

    if 'Tweet' in df.columns:
        st.write("### Preprocessing the Tweets...")
        df['Preprocessed_Tweet'] = df['Tweet'].apply(pre_processing)
        st.write("### Preprocessed Data:")
        st.write(df[['Tweet', 'Preprocessed_Tweet']].head())

        # Model selection
        st.write("### Choose a Model for Sentiment Analysis")
        model_choice = st.selectbox("Model", ["SVM (Count Vectorizer)", "SVM (TF-IDF Vectorizer)"])

        if model_choice == "SVM (Count Vectorizer)":
            model = svm_count
        elif model_choice == "SVM (TF-IDF Vectorizer)":
            model = svm_tfidf

        # Perform sentiment analysis
        st.write("### Predicting Sentiments...")
        df['Sentiment'] = model.predict(df['Preprocessed_Tweet'])
        st.write("### Results:")
        st.write(df[['Tweet', 'Sentiment']])

        # Download results
        st.download_button(
            label="Download Results",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="sentiment_results.csv",
            mime="text/csv",
        )

# Text input for single prediction
st.write("### Single Text Sentiment Analysis")
text_input = st.text_area("Enter text here:")
if text_input:
    preprocessed_text = pre_processing(text_input)
    st.write("Preprocessed Text:", preprocessed_text)

    model_choice = st.selectbox("Choose a Model", ["SVM (Count Vectorizer)", "SVM (TF-IDF Vectorizer)"], key="single_text")

    if model_choice == "SVM (Count Vectorizer)":
        model = svm_count
    elif model_choice == "SVM (TF-IDF Vectorizer)":
        model = svm_tfidf

    sentiment = model.predict([preprocessed_text])[0]
    st.write("Predicted Sentiment:", sentiment)
