
import re
from nltk.tokenize import TweetTokenizer
import string
import emoji
from nltk.stem import WordNetLemmatizer
from warnings import filterwarnings
filterwarnings('ignore')
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
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


tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
stop_words_set = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
punctuation = set(string.punctuation)


def removal_html(text):
    return re.sub(r'<.*?>','',text)
def removal_url(text):
    clean_text = re.sub(r'https?:\/\/[^\s]+','',text)
    return re.sub(r'www\.[a-z]?\.?(com)+|[a-z]*?\.?(com)+','',clean_text)
def twitter_handles(text):
    return re.sub(r'@\w*','',text)



def pre_processing(text):
    # Handle emojis
    text = emoji.demojize(text)

    # Replace slang and abbreviations
    text = ' '.join([slang_dic.get(word, word) for word in text.split()])

    # handling twitter handles
    text = twitter_handles(text)

    # removal of urls and www sites
    text = removal_url(text)      

    # removal of html tags       
    text = removal_html(text)            

    #Remove hastags but keep the text
    text = re.sub(r'#','',text)

    #Remove numbers but keep the text
    text = re.sub(r'\d+','',text)

    # Remove punctuation and stopwords
    text = ''.join([char if char not in punctuation else ' ' for char in text])
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words_set])

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    return text