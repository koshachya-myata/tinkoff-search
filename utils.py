import numpy as np

def build_invert_index(texts, index={}):
    for i, word in enumerate(texts):
        for w in set(word.split()):
            if w not in index:
                index[w] = []
            if i not in index[w]:
                index[w].append(i)
    return index

def intersect_2sets(set_a, set_b):
    # в моем случае они еще предварительно
    # отсортированы по рейтингу
    set1 = list(set_a)
    set2 = list(set_b)
    i = 0
    j = 0
    rt = []
    while i < len(set1) and j < len(set2):
        if set1[i] < set2[j]:
            i += 1
        elif set1[i] > set2[j]:
            j += 1
        else:
            rt.append(set1[i])
            i += 1
            j += 1
    return set(rt)

def precision_at_k(actual, predicted, k):
    # функцию я реализовал, но руками не размечал ничего
    # поэтому не пригодилась
    s = 0
    for el in predicted[:k]:
        if el in actual:
            s += 1
    return s / k