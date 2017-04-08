# Data files
TRAIN_FILE = "data/train.txt"
DEV_FILE = "data/dev.txt"
TEST_FILE = "data/test-hidden.txt"
PRETRAINED_EMBEDS_FILE = "data/pretrained-embeds.pkl"

DEV_GOLD = "data/dev.key"

class Actions:
    """Enum for each possible parser action"""
    SHIFT = 0
    REDUCE_L = 1
    REDUCE_R = 2

    NUM_ACTIONS = 3

    action_to_ix = { "SHIFT": SHIFT,
                     "REDUCE_L": REDUCE_L,
                     "REDUCE_R": REDUCE_R }

END_OF_INPUT_TOK = "<END-OF-INPUT>"
NULL_STACK_TOK = "<NULL-STACK>"
ROOT_TOK = "<ROOT>"
