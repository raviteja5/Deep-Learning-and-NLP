import operator
from collections import defaultdict, Counter
from gtnlplib.preproc import conll_seq_generator
from gtnlplib.constants import OFFSET, START_TAG, END_TAG

argmax = lambda x : max(x.iteritems(),key=operator.itemgetter(1))[0]

def get_tag_word_counts(filename):
    """build a dict of counters, one per tag, counting the words that go with each tag
    """
    all_counters = defaultdict(lambda : Counter())
    all_tags = set()
    for i,(words, tags) in enumerate(conll_seq_generator(filename,max_insts=100000)):
        for word,tag in zip(words,tags):
            all_counters[tag][word] += 1
    return all_counters

def get_most_common_word_weights(trainfile):
    """Return a set of weights, so that each word is tagged by its most frequent tag in the training file.
    If word doesn't appear, set its output tag as Noun in the weights.
    """
    weights = defaultdict(float)
    counters = get_tag_word_counts(trainfile)
    words = set()
    for tag,tag_ctr in counters.iteritems():
        words = words.union(set(tag_ctr))
    for word in words:
        maxw = 1.
        curtag = 'NOUN'
        for tag,tag_ctr in counters.iteritems():
            if tag_ctr[word] > maxw:
                curtag = tag
                maxw = tag_ctr[word]
        weights[(curtag,word)] = maxw
    return weights
