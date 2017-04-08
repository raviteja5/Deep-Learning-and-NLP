import utils
import torch
import torch.nn as nn
import torch.autograd as ag
import torch.nn.functional as F

from gtnlplib.constants import Actions

# INITIAL WORD EMBEDDING COMPONENTS

class VanillaWordEmbeddingLookup(nn.Module):
    """
    A component that simply returns a list of the word embeddings as
    autograd Variables.
    """

    def __init__(self, word_to_ix, embedding_dim):
        super(VanillaWordEmbeddingLookup, self).__init__()
        self.word_to_ix = word_to_ix
        self.embedding_dim = embedding_dim
        self.use_cuda = False

        self.output_dim = embedding_dim
        self.word_embeddings = nn.Embedding(len(word_to_ix),embedding_dim)


    def forward(self, sentence):
        inp = utils.sequence_to_variable(sentence, self.word_to_ix)
        embeds = [] 
        for v in inp:
            embeds.append(self.word_embeddings(v))
        return embeds


class BiLSTMWordEmbeddingLookup(nn.Module):
    """
    A component that uses a Bidirectional LSTM to initialize the word embeddings before
    parsing begins.
    """
    def __init__(self, word_to_ix, word_embedding_dim, hidden_dim, num_layers, dropout):
        super(BiLSTMWordEmbeddingLookup, self).__init__()
        self.word_to_ix = word_to_ix
        self.num_layers = num_layers
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = False

        self.output_dim = hidden_dim

        self.word_embeddings = nn.Embedding(len(word_to_ix),word_embedding_dim)
        self.lstm = nn.LSTM(word_embedding_dim,hidden_dim/2,num_layers=num_layers, bidirectional=True, dropout=dropout)

        self.hidden = self.init_hidden()

    def forward(self, sentence):
        inp = utils.sequence_to_variable(sentence, self.word_to_ix, self.use_cuda)
        
        idxs = map(lambda w: self.word_to_ix[w], sentence)
        tensor = torch.LongTensor(idxs)
        inputs = ag.Variable(tensor)
        embeds = self.word_embeddings(inputs)
        
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence),1,-1),self.hidden)
        return [i for i in lstm_out]

    def init_hidden(self):
        return (ag.Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim/2)),
                ag.Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim/2)))

    def clear_hidden_state(self):
        self.hidden = self.init_hidden()



# COMBINER NETWORK COMPONENTS

class MLPCombinerNetwork(nn.Module):
    """
    An MLP network which takes the top 2 elements in the parser stack,
    combines them, and puts the new embedding back on the stack.    
    """

    def __init__(self, embedding_dim):
        super(MLPCombinerNetwork, self).__init__()
        self.linear1 = nn.Linear(2*embedding_dim,embedding_dim)
        self.linear2 = nn.Linear(embedding_dim,embedding_dim)

    def forward(self, head_embed, modifier_embed):
        embeds = utils.concat_and_flatten([head_embed,modifier_embed])
        out = F.tanh(self.linear1(embeds))
        new_embeddings = self.linear2(out)
        return new_embeddings

class LSTMCombinerNetwork(nn.Module):
    """
    This serves the same purpose as the previous one, but uses an LSTM instead.
    Every action to produce a new embedding corresponds to one time step of the LSTM.
    """

    def __init__(self, embedding_dim, num_layers, dropout):
        super(LSTMCombinerNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(2*embedding_dim,embedding_dim,num_layers=num_layers, dropout=dropout)

        self.hidden = self.init_hidden()


    def init_hidden(self):
        return (ag.Variable(torch.FloatTensor(self.num_layers, 1, self.embedding_dim).zero_()),
                ag.Variable(torch.FloatTensor(self.num_layers, 1, self.embedding_dim).zero_()))


    def forward(self, head_embed, modifier_embed):
        embeds = utils.concat_and_flatten([head_embed,modifier_embed])
        lstm_out, self.hidden = self.lstm(embeds.view(1,1,-1),self.hidden)
        
        return lstm_out[0]

    def clear_hidden_state(self):
        self.hidden = self.init_hidden()

#ACTION CHOOSING COMPONENT

class ActionChooserNetwork(nn.Module):
    """
    This network takes the features from the feature extractor,
    and decides the next action that the parser has to take. It 
    can be one among : Shift, Reduce Left and Reduce Right
    """

    def __init__(self, input_dim):
        super(ActionChooserNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim,input_dim)
        self.linear2 = nn.Linear(input_dim,3)

    def forward(self, inputs):
        embeds = utils.concat_and_flatten(inputs)
        out = F.relu(self.linear1(embeds))
        log_probs = F.log_softmax(self.linear2(out))
        return log_probs
