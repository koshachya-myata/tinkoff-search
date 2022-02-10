import numpy as np
import swifter
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances

from utils import intersect_2sets, precision_at_k, build_invert_index
from process import preproccessing

class Document:
    def __init__(self, title, text, rating, text_processed, title_processed):
        self.title = title
        self.text = text
        self.rating = rating
        self.text_processed = text_processed
        self.title_processed = title_processed
    
    def format(self, query):
        return [self.title, self.text[:85] + ' ...']

index = []
inv_index = {}

f = open('pickled', 'rb')
data = pickle.load(f)
vectorizer = data['text_model']
vectorizer_title = data['title_model']
f.close()

def build_index():
    global index
    global inv_index

    for i in range(len(data['titles'])):
        index.append(Document(title=data['titles'][i], text=data['texts'][i], 
            rating=data['ratings'][i],
            text_processed=data['texts_processed'][i],
            title_processed=data['titles_processed'][i]))
    # обратный индекс строится долговато
    inv_index = build_invert_index(data['texts_processed'])
    inv_index = build_invert_index(data['titles_processed'], inv_index)
    #for w in inv_index.keys():  # я вообще их в ноутбуке сортирую. Тут просто еще раз
    #                            # чтобы и для не сорт. данных работало.
    #    inv_index[w].sort(key=lambda ind: index[ind].rating, reverse=True)

def score(query, document):
    q_text_model = np.squeeze(vectorizer.transform([preproccessing(query.lower()).rstrip()]).todense())
    q_title_model = np.squeeze(vectorizer_title.transform([preproccessing(query.lower()).rstrip()]).todense())
    d_text_model = np.squeeze(vectorizer.transform([document.text_processed]).todense())
    d_title_model = np.squeeze(vectorizer_title.transform([document.title_processed]).todense())
    c1 = cosine_similarity(d_text_model, q_text_model)[0][0]
    c2 = cosine_similarity(d_title_model, q_title_model)[0][0]
    # можно было бы обучить лин. рег., но для этого
    # нужно руками разметить какое-нибудь число данных
    # а я не очень хочу это делать
    return 0.5*c1 + 0.5*c2


def retrieve(query):
    if query == '':
        return index[:30]
    q = preproccessing(query.lower())
    kw = q.split()
    if len(kw) == 0 or kw[0] not in inv_index.keys():
        return []
    rt = inv_index[kw[0]]
    for w in kw[1:]:
        if w not in inv_index.keys():
            return []
        rt = intersect_2sets(rt, inv_index[w])
    return [index[d] for d in list(rt)[:1000]] # я там в сервере потом отрезаю до 30, уже по скору, а не только по рейтингу
