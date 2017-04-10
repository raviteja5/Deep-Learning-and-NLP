import operator
from collections import defaultdict, Counter
from gtnlplib.constants import START_TAG,END_TAG
import numpy as np

def argmax(scores):
    """Find the key that has the highest value in the scores dict"""
    return max(scores.iteritems(),key=operator.itemgetter(1))[0]

def viterbi_step(tag, m, words, feat_func, weights, prev_scores):
    """
    Calculate the best path score and back pointer for a given node in the trellis
    """
    feats = dict()
    scores = dict() 
    for key in prev_scores:
        temp = prev_scores[key]
        feats = feat_func(words,tag,key,m)
        for f in feats:
            if f in weights:
                temp+=weights[f]
        scores[key]=temp
    best_score = max(scores.values())
    best_tag = argmax(scores)
    return best_score, best_tag

def build_trellis(tokens,feat_func,weights,all_tags):
    """Construct a trellis for the hidden Markov model. Output is a list of dicts.
    """    
    trellis = [None]*(len(tokens))

    # build the first column separately
    trellis[0] = dict()
    for tag in all_tags:
        temp = 0.0
        feats = feat_func(tokens,tag,START_TAG,0)
        for f in feats:
            if f in weights:
                temp+=weights[f]
        trellis[0][tag] = (temp,START_TAG)
    
    # iterate over the remaining columns
    for m in range(1,len(tokens)):
        trellis[m] = dict()
        for tag in all_tags:
            temp=dict()
            for t in trellis[m-1]:
                temp[t]=trellis[m-1][t][0]
            trellis[m][tag] = viterbi_step(tag, m, tokens, feat_func, weights, temp)
        
    return trellis


def viterbi_tagger(tokens,feat_func,weights,all_tags):
    """Tag the given words using the viterbi algorithm
    """
    trellis = build_trellis(tokens,feat_func,weights,all_tags)

    # Find last tag and best score
    final_scores = dict()
    for tag in all_tags:
        feats = feat_func(tokens,END_TAG,tag,len(tokens))
        final_scores[tag] = trellis[len(tokens)-1][tag][0] + weights[feats.keys()[0]]
    last_tag = argmax(final_scores)
    best_score = max(final_scores.values())

    # Walk backwards through trellis to find best tag sequence
    output = [last_tag] 
    for m,v_m in enumerate(reversed(trellis[1:])): 
        temp = v_m[last_tag]
        add = temp[1]
        output = [add] + output
        last_tag = add

    return output,best_score

