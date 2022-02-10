from nltk.tokenize import word_tokenize, sent_tokenize
import re
import string
import nltk
from nltk.corpus import stopwords
import pymorphy2

#nltk.download('stopwords')
stopWords = set(stopwords.words("russian"))
m = pymorphy2.MorphAnalyzer(lang='ru')

def remove_numbers(text):
    return ''.join([i if not i.isdigit() else ' ' for i in text])

def remove_punctuation(text):
    punc = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~»«—–-"
    return "".join([ch if ch not in punc else ' ' for ch in text])

def remove_multiple_spaces(text):
    return re.sub(r'\s+', ' ', text, flags=re.I)

def get_normal_form(text):
    return ' '.join([m.parse(w)[0].normal_form for w in word_tokenize(text)])

def delete_stopwords(text):
    return ' '.join([(word) for word in word_tokenize(text) if word not in stopWords])

def preproccessing(text):
    return remove_multiple_spaces(delete_stopwords(get_normal_form(remove_numbers(remove_punctuation(text.strip())))))