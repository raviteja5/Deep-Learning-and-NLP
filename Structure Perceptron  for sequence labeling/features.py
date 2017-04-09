from gtnlplib import constants

def word_feats(words,y,y_prev,m):
    """
    Current word feature, and offset
    """
    fv = dict()
    if m < len(words):
        fv[(y,constants.CURR_WORD_FEAT,words[m])] = 1
    fv[(y,constants.OFFSET)] = 1
    return fv

def word_suff_feats(words,y,y_prev,m):
    """
    Same as before, with an additional suffix feature
    """
    feats = word_feats(words,y,y_prev,m)
    feats[(y,constants.SUFFIX_FEAT,words[m][len(words[m])-2:])] = 1
    return feats
    
def word_neighbor_feats(words,y,y_prev,m):
    """
    Additionally include features of previous and next words
    """
    feats = word_feats(words,y,y_prev,m)
    feats[(y,constants.PREV_WORD_FEAT,words[m-1] if m>0 else constants.PRE_START_TOKEN)] = 1
    if m<len(words):
        feats[(y,constants.NEXT_WORD_FEAT,words[m+1] if m<len(words)-1 else constants.POST_END_TOKEN)] = 1
    return feats
    
def word_feats_competitive_en(words,y,y_prev,m):
    feats = word_feats(words,y,y_prev,m)
    feats[(y,constants.SUFFIX_FEAT,words[m][len(words[m])-2:])] = 1
    feats[(y,constants.PREV_WORD_FEAT,words[m-1] if m>0 else constants.PRE_START_TOKEN)] = 1
    if m<len(words):
        feats[(y,constants.NEXT_WORD_FEAT,words[m+1] if m<len(words)-1 else constants.POST_END_TOKEN)] = 1
    
    feats[(y,"PREFIX_FEAT",words[m][:2])] = 1
    feats[(y,constants.PREV_TAG_FEAT,y_prev)]=1
    return feats
    
def word_feats_competitive_ja(words,y,y_prev,m):
    feats = word_feats(words,y,y_prev,m)
    feats[(y,constants.SUFFIX_FEAT,words[m][len(words[m])-2:])] = 1
    feats[(y,constants.PREV_WORD_FEAT,words[m-1] if m>0 else constants.PRE_START_TOKEN)] = 1
    if m<len(words):
        feats[(y,constants.NEXT_WORD_FEAT,words[m+1] if m<len(words)-1 else constants.POST_END_TOKEN)] = 1
    
    feats[(y,"PREFIX_FEAT",words[m][:2])] = 1
    feats[(y,constants.PREV_TAG_FEAT,y_prev)]=1
    return feats

def hmm_feats(words,y,y_prev,m):
    result = dict()
    if(m<len(words)):
        result[(y,constants.CURR_WORD_FEAT,words[m])] = 1
    result[(y,constants.PREV_TAG_FEAT,y_prev)]=1
    return result
