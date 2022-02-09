import numpy as np
import pandas as pd
import multiprocessing as mp
import swifter
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem
from nltk.stem.snowball import SnowballStemmer
import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
import pickle
from utils import preproccessing
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances

f = open('pickled', 'rb')
data = pickle.load(f)
vectorizer = data['text_model']
f.close()

class Document:
    def __init__(self, title, text, rating, text_vec, text_processed, title_processed):
        # можете здесь какие-нибудь свои поля подобавлять
        self.title = title
        self.text = text
        self.rating = rating
        self.text_vec = text_vec
        self.text_processed = text_processed
        self.title_processed = title_processed

    
    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title, self.text[:85] + ' ...']

index = []

index2 = {}
def build_index():
    # считывает сырые данные и строит индекс
    for i in range(len(data['titles'])):
        index.append(Document(title=data['titles'][i], text=data['texts'][i], 
            rating=data['ratings'][i], text_vec=data['texts_vec'][i],
            text_processed=data['texts_processed'][i],
            title_processed=data['titles_processed'][i]))

    for i in range(len(data['titles'])):
        for word in set(data['texts_processed']):
            if word not in index2:
                index2[word] = []
            index2[word].append(i)
        for word in set(data['titles_processed']):
            if word not in index2:
                index2[word] = []
            index2[word].append(i)

def score(query, document):
    # возвращает какой-то скор для пары запрос-документ
    # больше -- релевантнее
    vec1 = document.text_vec
    vec2 = np.squeeze(vectorizer.transform([preproccessing(query.lower()).rstrip()]).todense())
    return cosine_similarity(vec1, vec2)[0][0]
    return random.random()

def retrieve(query):
    print(query)
    # возвращает начальный список релевантных документов
    # (желательно, не бесконечный)
    if query == '' or query:
        candidates = []
        for doc in index:
            if query.lower() in doc.title.lower() or query in doc.text.lower():
                candidates.append(doc)
        return candidates[:50]

    kw = query.split()
    s = set(index2[kw[0]])
    for w in kw[1:]:
        s.intersect(index2[word])
    return list(s)
