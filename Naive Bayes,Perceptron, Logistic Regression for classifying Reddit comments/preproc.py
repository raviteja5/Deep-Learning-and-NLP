import nltk
import pandas as pd
from collections import Counter

def tokenize_and_downcase(string,vocab=None):
    sentences = nltk.sent_tokenize(string)
    bow = Counter()
    for s in sentences:
        words = nltk.word_tokenize(s)
        words_small = [w.lower() for w in words]
        bow.update(words_small)
    return bow

### Helper code

def get_corpus_counts(list_of_bags_of_words):
    counts = Counter()
    for bow in list_of_bags_of_words:
        for key,val in bow.iteritems():
            counts[key] += val
    return counts

def custom_preproc(string):
    """remove the special characters for better accuracy
    """
    bow = tokenize_and_downcase(string)
    special_char = [',','!',';','.','?','\'',':','/','s']
    for c in special_char:
        bow[c]=0
    return bow
