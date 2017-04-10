from gtnlplib.constants import OFFSET
import numpy as np

argmax = lambda x : max(x.iteritems(),key=lambda y : y[1])[0]

def make_feature_vector(base_features,label):
    """take a counter of base features and a label; return a dict of features, corresponding to f(x,y)
    """
    result = {(label,x):base_features[x] for x in base_features}
    result[(label,'**OFFSET**')] = 1
    return result
    
def predict(base_features,weights,labels):
    """prediction function
    For Naive Bayes, computes P(X,Y) = P(X|Y) * P(Y) 
    (first term is over entire sentence, second term is taken care by offset)
    """
    scores = dict()
    for label in labels:
        label_weights = {(x,y):weights[(x,y)] for (x,y) in weights if x==label}
        temp_score = 0.0
        for (x,y) in label_weights:
            if(y in base_features and base_features[y] > 0):
                temp_score += float(base_features[y]) * label_weights[(x,y)]
            elif y == '**OFFSET**':
                temp_score += 1.0 * label_weights[(x,y)]
        scores[label] = temp_score
    return argmax(scores),scores

def predict_all(x,weights,labels):
    """Predict the label for all instances in a dataset
    """
    y_hat = np.array([predict(x_i,weights,labels)[0] for x_i in x])
    return y_hat

def get_top_features_for_label(weights,label,k=5):
    """Return the five features with the highest weight for a given label.
    """
    label_weights = [((x,y),weights[(x,y)]) for (x,y) in weights if x==label]
    return sorted(label_weights, key=lambda w: -w[1])[:k]
