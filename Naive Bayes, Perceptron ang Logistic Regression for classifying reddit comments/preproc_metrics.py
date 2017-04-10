def get_token_type_ratio(vocabulary):
    """compute the ratio of tokens to types
    """
    return float(sum(vocabulary.values())) / len(list(vocabulary))

def type_frequency(vocabulary, k):
    """compute the number of words that occur exactly k times
    """
    return sum([1 if x==k else 0 for x in vocabulary.values()])

def unseen_types(first_vocab, second_vocab):
    """compute the number of words that appear in the second vocab but not in the first vocab
    """
    return len(set(second_vocab).difference(set(first_vocab)))
    
