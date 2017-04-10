from collections import defaultdict
from gtnlplib.clf_base import predict,make_feature_vector,argmax

def perceptron_update(x,y,weights,labels):
    """compute the perceptron update for a single instance
    """
    update = defaultdict(float, {})
    predictedLabel,scores = predict(x,weights,labels)
    if predictedLabel != y:
        update.update(make_feature_vector(x,y))
        temp = make_feature_vector(x,predictedLabel)
        for x,y in temp.iteritems():
            temp[x] = -y
        update.update(temp)
    return update


def estimate_perceptron(x,y,N_its):
    """estimate perceptron weights for N_its iterations over the dataset (x,y)

    """
    labels = set(y)
    weights = defaultdict(float)
    weight_history = []
    for it in xrange(N_its):
        for x_i,y_i in zip(x,y):
            for s,t in perceptron_update(x_i,y_i,weights,labels).iteritems():
                weights[s] += t
        weight_history.append(weights.copy())
    return weights, weight_history

def estimate_avg_perceptron(x,y,N_its):
    """estimate averaged perceptron classifier
    """
    labels = set(y)
    weights = defaultdict(float)
    weight_history = []
    avg_weights = defaultdict()
    
    for it in xrange(N_its):
        w_sum = defaultdict(float) 
        t=0.0 
        for x_i,y_i in zip(x,y):
            update = perceptron_update(x_i,y_i,weights,labels)
            if len(update)>0:
                for m,n in update.iteritems():
                    weights[m] += n
                    w_sum[m] += (t)*n
                t=t+1
        #Compute avg weight
        avg_weights.clear()
        temp = dict()
        for f in weights:
            avg_weights[f] = weights[f] - ( float(w_sum[f])/t if f in w_sum else 0) 
        weight_history.append(avg_weights.copy())
        weights = defaultdict(float,avg_weights.copy())
    return avg_weights, weight_history
