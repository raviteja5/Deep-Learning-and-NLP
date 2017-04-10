from gtnlplib.preproc import conll_seq_generator
from gtnlplib.constants import START_TAG, TRANS, END_TAG, EMIT, OFFSET
from gtnlplib import naive_bayes, most_common
import numpy as np
from collections import defaultdict

def hmm_features(tokens,curr_tag,prev_tag,m):
    """Feature function for HMM that returns emit and transition features
    """
    result = dict()
    if(m<len(tokens)):
        result[(curr_tag,tokens[m],EMIT)] = 1
    result[(curr_tag,prev_tag,TRANS)]=1
    return result
    

def compute_HMM_weights(trainfile,smoothing):
    """Compute all weights for the HMM
    """
    tag_trans_counts = most_common.get_tag_trans_counts(trainfile)
    all_tags = tag_trans_counts.keys()
    
    trans_wts = compute_transition_weights(tag_trans_counts, smoothing)
    counters = most_common.get_tag_word_counts(trainfile)
    theta_nb = naive_bayes.estimate_nb_tagger(counters,smoothing)
    
    result = defaultdict(float)
    for tag,word in theta_nb:
        result[tag,word,EMIT] = theta_nb[tag,word]            
    result.update(trans_wts)

    return result,all_tags


def compute_transition_weights(trans_counts, smoothing):
    """Computes the HMM transition weights, given the counts.
    """
    weights = defaultdict(float)
    tags = trans_counts.keys()
    for prev_tag in tags:
        cur_ctr = trans_counts[prev_tag]
        total = sum(cur_ctr.values())
        dm = float(total + (smoothing * (len(tags)-1)))  #-1 for not including start tag
        for cur_tag in tags+[END_TAG]:
            if prev_tag == END_TAG:
                weights[(cur_tag,prev_tag,TRANS)] = -np.inf
                
            if cur_tag == START_TAG:
                weights[(cur_tag,prev_tag,TRANS)] = -np.inf
            elif cur_tag in cur_ctr:
                weights[(cur_tag,prev_tag,TRANS)] = np.log ( (float(cur_ctr[cur_tag]) + smoothing) / dm)
            else:
                weights[(cur_tag,prev_tag,TRANS)] = np.log(float(smoothing) / float(dm))
    for tag in tags+[END_TAG]:
        weights[(tag,END_TAG,TRANS)] = -np.inf
        weights[(START_TAG,tag,TRANS)] = -np.inf
    return weights