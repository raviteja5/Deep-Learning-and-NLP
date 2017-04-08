from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.autograd as ag

from gtnlplib.constants import Actions, NULL_STACK_TOK, END_OF_INPUT_TOK, ROOT_TOK
import gtnlplib.utils as utils
import gtnlplib.neural_net as neural_net

DepGraphEdge = namedtuple("DepGraphEdge", ["head", "modifier"])
StackEntry = namedtuple("StackEntry", ["headword", "headword_pos", "embedding"])


class ParserState:
    """
    Stores the state of the stack parser and input buffer, and provides methods
    to operate on them
    """
    def __init__(self, sentence, sentence_embs, combiner, null_stack_tok_embed=None):
        self.combiner = combiner

        # Pointer to next element in the input buffer
        self.curr_input_buff_idx = 0
        self.input_buffer = [ StackEntry(we[0], pos, we[1]) for pos, we in enumerate(zip(sentence, sentence_embs)) ]

        self.stack = []
        self.null_stack_tok_embed = null_stack_tok_embed

    def shift(self):
        #if self.curr_input_buff_idx >= len(self.input_buffer)-1:
        #    print "Error:",self.__str__()
        next_item = self.input_buffer[self.curr_input_buff_idx]
        self.stack.append(next_item)
        self.curr_input_buff_idx += 1

    def reduce_left(self):
        return self._reduce(Actions.REDUCE_L)

    def reduce_right(self):
        return self._reduce(Actions.REDUCE_R)

    def done_parsing(self):
        return (self.input_buffer_len() == 1 ) and (self.stack_len()==1)   

    def stack_len(self):
        return len(self.stack)

    def input_buffer_len(self):
        return len(self.input_buffer) - self.curr_input_buff_idx

    def stack_peek_n(self, n):
        #Append Null stack token if request size is more than what is present
        if len(self.stack) - n < 0:
            return [ StackEntry(NULL_STACK_TOK, -1, self.null_stack_tok_embed) ] * (n - len(self.stack)) \
                   + self.stack[:]
        return self.stack[-n:]

    def input_buffer_peek_n(self, n):
        assert self.curr_input_buff_idx + n - 1 <= len(self.input_buffer)
        return self.input_buffer[self.curr_input_buff_idx:self.curr_input_buff_idx+n]

    def _reduce(self, action):
        """
        Reduce the top 2 elements of the stack into a single element,
        based on the provided action.
        """
        assert len(self.stack) >= 2
        right = self.stack.pop()
        left = self.stack.pop()
        head = right if action == Actions.REDUCE_L else left
        mod = left if action == Actions.REDUCE_L else right
        self.stack.append( StackEntry(head.headword, head.headword_pos, self.combiner(head.embedding,mod.embedding)) )
        return DepGraphEdge((head.headword, head.headword_pos),(mod.headword, mod.headword_pos))

class TransitionParser(nn.Module):
    """
    Contains the core parsing logic.
    """
    def __init__(self, feature_extractor, word_embedding_component, action_chooser_component, combiner_component):
        super(TransitionParser, self).__init__()

        self.word_embedding_component = word_embedding_component
        self.feature_extractor = feature_extractor
        self.combiner = combiner_component
        self.action_chooser = action_chooser_component
        self.use_cuda = False

        self.null_stack_tok_embed = nn.Parameter(torch.randn(1, word_embedding_component.output_dim))


    def forward(self, sentence, actions=None):
        self.refresh() 

        padded_sent = sentence + [END_OF_INPUT_TOK]

        # Initialize the parser state
        sentence_embs = self.word_embedding_component(padded_sent)

        parser_state = ParserState(padded_sent, sentence_embs, self.combiner, null_stack_tok_embed=self.null_stack_tok_embed)
        outputs = [] # Holds the output of each action decision
        actions_done = [] # Holds all actions done so far

        dep_graph = set()

        # Make the action queue if we have it
        if actions is not None:
            action_queue = deque()
            action_queue.extend([ Actions.action_to_ix[a] for a in actions ])
            have_gold_actions = True
        else:
            have_gold_actions = False

        while(not parser_state.done_parsing()):
            feats = self.feature_extractor.get_features(parser_state)
            log_probs = self.action_chooser(feats)
            outputs.append(log_probs)
            todo = utils.argmax(log_probs) if (not have_gold_actions) else action_queue.popleft()
            if todo == 0 and parser_state.input_buffer_len() > 1 :
                parser_state.shift()
                actions_done.append(Actions.SHIFT)
            elif todo == 1 and (parser_state.stack_len()>1):
                dep_edge = parser_state.reduce_left()
                dep_graph.add(dep_edge)
                actions_done.append(Actions.REDUCE_L)

            else:
                dep_edge = parser_state.reduce_right()
                dep_graph.add(dep_edge)
                actions_done.append(Actions.REDUCE_R)
            
        dep_graph.add(DepGraphEdge((ROOT_TOK, -1), (parser_state.stack[-1].headword, parser_state.stack[-1].headword_pos)))
        return outputs, dep_graph, actions_done


    def refresh(self):
        if isinstance(self.combiner, neural_net.LSTMCombinerNetwork):
            self.combiner.clear_hidden_state()
        if isinstance(self.word_embedding_component, neural_net.BiLSTMWordEmbeddingLookup):
            self.word_embedding_component.clear_hidden_state()


    def predict(self, sentence):
        _, dep_graph, _ = self.forward(sentence)
        return dep_graph


    def predict_actions(self, sentence):
        _, _, actions_done = self.forward(sentence)
        return actions_done

    def to_cpu(self):
        self.use_cuda = False
        self.word_embedding_component.use_cuda = False
        self.combiner.use_cuda = False
        self.cpu()


def train(data, model, optimizer):

    correct_actions = 0
    total_actions = 0
    tot_loss = 0.
    instance_count = 0

    for sentence, actions in data:

        if len(sentence) <= 2:
            continue

        optimizer.zero_grad()
        model.refresh()

        outputs, _, actions_done = model(sentence, actions)

        loss = ag.Variable(torch.FloatTensor([0]))
        action_idxs = [ ag.Variable(torch.LongTensor([ a ])) for a in actions_done ]

        for output, act in zip(outputs, action_idxs):
            loss += nn.NLLLoss(output.view(-1, 3), act)

        tot_loss += utils.to_scalar(loss.data)
        instance_count += 1

        for gold, output in zip(actions_done, outputs):
            pred_act = utils.argmax(output.data)
            if pred_act == gold:
                correct_actions += 1
        total_actions += len(outputs)

        loss.backward()
        optimizer.step()

    acc = float(correct_actions) / total_actions
    loss = float(tot_loss) / instance_count
    print "Number of instances: {}    Number of network actions: {}".format(instance_count, total_actions)
    print "Acc: {}  Loss: {}".format(float(correct_actions) / total_actions, tot_loss / instance_count)
