from gtnlplib.preproc import get_corpus_counts
from gtnlplib.constants import OFFSET
from gtnlplib import clf_base, evaluation

import numpy as np
from collections import defaultdict, Counter

def get_corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.
    """
    
    t = Counter()
    for bow in range(len(x)):
        if y[bow] == label:
            for key,val in x[bow].iteritems():
                t[key] += float(val)
    return defaultdict(float,dict(t))
    
def estimate_pxy(x,y,label,smoothing,vocab):
    """Compute smoothed log-probability P(word | label) for a given label.
    """
    result = defaultdict(float)
    counts = get_corpus_counts(x,y,label)
    dm = sum(counts.values()) + float(len(vocab)*smoothing)
    for w in vocab:
        nm = float(counts[w]) + smoothing
        result[w] = np.log(nm/dm)
    return result
    
def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model
    Computes P(word|label) for each (word,label) pair
    """
    num_counts = Counter()
    for bow in x:
        for key,val in bow.iteritems():
            num_counts[key] += val
    keys = dict(num_counts).keys()
    result = defaultdict(float)
    labels = set(y)
    doc_counts = defaultdict(float)
    # Ratio of docs with a given label
    for label in labels:
        num_docs = float(sum(l == label for l in y))
        doc_counts[label] = np.log(num_docs / len(y))
        result[(label,OFFSET)] = doc_counts[label]
    # For each label, compute P(x,y) 
    for label in labels:
        pxy = estimate_pxy(x,y,label,smoothing, keys)
        for w in keys:
            result[(label,w)] = pxy[w]     
    return result
    
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    """find the smoothing value that gives the best accuracy on the dev data
    """
    scores={}
    labels = set(y_tr)
    for s in smoothers:
        theta_nb = estimate_nb(x_tr,y_tr,s)
        y_hat = clf_base.predict_all(x_dv,theta_nb,labels)
        scores[s]=evaluation.acc(y_hat,y_dv)
    l = scores.values()
    best = smoothers[np.argmax(l)]
    return best,scores
