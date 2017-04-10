import numpy as np 
import sys
from collections import defaultdict,Counter
from gtnlplib import scorer, most_common,preproc
from gtnlplib.constants import OFFSET

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
    """
    num_counts = Counter()
    for bow in x:
        for key,val in bow.iteritems():
            num_counts[key] += val
    keys = dict(num_counts).keys()
    result = defaultdict(float)
    labels = set(y)
    doc_counts = defaultdict(float)
    for label in labels:
        num_docs = float(sum(l == label for l in y))
        doc_counts[label] = np.log(num_docs / len(y))
        result[(label,OFFSET)] = doc_counts[label]
    for label in labels:
        pxy = estimate_pxy(x,y,label,smoothing, keys)
        for w in keys:
            result[(label,w)] = pxy[w]     
    return result

def estimate_nb_tagger(counters,smoothing):
    """build a tagger based on the naive bayes classifier, which correctly accounts for the prior P(Y)
    """
    sorted_tags = sorted(counters.keys())
    nb_weights = estimate_nb([counters[tag] for tag in sorted_tags],
                                     sorted_tags,
                                     .01)
    total = sum([sum(counters[tag].itervalues()) for tag in sorted_tags])
    for tag in counters: 
        cur = sum(counters[tag].itervalues()) + 0.0
        nb_weights[tag,OFFSET] = np.log(cur/total)
    return nb_weights
    
    
    
