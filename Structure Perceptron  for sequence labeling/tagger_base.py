from gtnlplib import scorer
from gtnlplib import preproc
from gtnlplib.constants import DEV_FILE, OFFSET, TRAIN_FILE, START_TAG, END_TAG

import operator
from collections import defaultdict
import tempfile
import matplotlib.pyplot as plt

argmax = lambda x : max(x.iteritems(),key=lambda y : y[1])[0]

def classifier_tagger(tokens,feat_func,weights,all_tags):
    """Tags a sequence of tokens, using the given feature function and list of weights
    """
    
    ptags = []
    fin_scores=[]
    for i in range(len(tokens)):
        scores = dict()
        for tag in all_tags:
            features = feat_func(tokens,tag,"IGNORE",i)
            temp_score = 0.0
            for f in features:
                if f in weights:
                    temp_score += float(weights[f]) * features[f]
            scores[tag] = temp_score
        ptags.append(argmax(scores))
        fin_scores.append(max(scores.values()))
    return ptags,sum(fin_scores)

def compute_features(tokens,tags,feat_func):
    """Compute dict of features and counts for a token and tag sequence
    """
    feats = dict()
    tags.append(END_TAG)
    M = len(tokens) +1
    for i in range(M):
        prev_tag = START_TAG if i==0 else tags[i-1]
        t_feats = feat_func(tokens,tags[i],prev_tag,i)
        for f in t_feats:
            if f in feats:
                feats[f] += t_feats[f]
            else:
                feats[f] = float(t_feats[f])
    return feats
    
