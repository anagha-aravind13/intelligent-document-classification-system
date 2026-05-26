import re
from matplotlib import text
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def preprocess_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    tokens = word_tokenize(text)

    filtered_tokens = [
        word
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(filtered_tokens)