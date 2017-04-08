import torch
import torch.autograd as ag
from gtnlplib.constants import END_OF_INPUT_TOK


def word_to_variable_embed(word, word_embeddings, word_to_ix):
    return word_embeddings(ag.Variable(torch.LongTensor([ word_to_ix[word] ])))


def sequence_to_variable(sequence, to_ix):
    return ag.Variable( torch.LongTensor([ to_ix[t] for t in sequence ]) )


def to_scalar(var):
    if isinstance(var, ag.Variable):
        return var.data.view(-1).tolist()[0]
    else:
        return var.view(-1).tolist()[0]

def argmax(vector):
    _, idx = torch.max(vector, 1)
    return to_scalar(idx)


def concat_and_flatten(items):
    return torch.cat(items, 1).view(1, -1)


def initialize_with_pretrained(pretrained_embeds, word_embedding_component):
    for word, index in word_embedding_component.word_to_ix.items():
        if word in pretrained_embeds:
            word_embedding_component.word_embeddings.weight.data[index] = torch.Tensor(pretrained_embeds[word])

