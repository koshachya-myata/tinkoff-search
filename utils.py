from nltk.tokenize import word_tokenize, sent_tokenize
from pymystem3 import Mystem
from nltk.stem.snowball import SnowballStemmer
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

wnl = nltk.WordNetLemmatizer()
stopWords = set(stopwords.words("russian"))
mystem = Mystem() 
stemmer = SnowballStemmer(language="russian")
def remove_numbers(text):
    return ''.join([i if not i.isdigit() else ' ' for i in text])

def remove_punctuation(text):
    punc = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~»«—–-"
    return "".join([ch if ch not in punc else ' ' for ch in text])

def remove_multiple_spaces(text):
    return re.sub(r'\s+', ' ', text, flags=re.I)

def lemmatize(text):
    return ''.join(mystem.lemmatize(text.lower()))

def stem(text):
    return ''.join([stemmer.stem(word) for word in text])

def delete_stopwords(text):
    return ' '.join([(word) for word in word_tokenize(text) if word not in stopWords])

def preproccessing2(text):
    return remove_multiple_spaces((remove_punctuation(delete_stopwords(remove_numbers(lemmatize(text.rstrip()))))))

def preproccessing(text):
    return remove_multiple_spaces(delete_stopwords(stem(remove_numbers(remove_punctuation(text.strip())))))