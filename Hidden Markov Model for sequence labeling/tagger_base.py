from gtnlplib import scorer
from gtnlplib import preproc
from gtnlplib import clf_base 
from gtnlplib.constants import DEV_FILE, OFFSET, TRAIN_FILE
import operator

def make_classifier_tagger(weights):
    def classify(words, all_tags):
        """Returns a list of tags, computed using a classifier with the weights passed as arguments to make_classifier_tagger
        """
        return [clf_base.predict({word:1},weights,all_tags)[0] for word in words]
    return classify
