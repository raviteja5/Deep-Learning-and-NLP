from gtnlplib import tagger_base, constants
from collections import defaultdict

def sp_update(tokens,tags,weights,feat_func,tagger,all_tags):
    """
    Computes the update for structure perceptron for a single instance
    """
    update = defaultdict(float, {})
    predicted_tags, score = tagger(tokens,feat_func,weights,all_tags)
    for i in range(len(tags)):        
        if predicted_tags[i] != tags[i]:
            old_features = feat_func(tokens,predicted_tags[i],predicted_tags[i-1] if i>0 else constants.START_TAG,i)
            new_features = feat_func(tokens,tags[i],tags[i-1] if i>0 else constants.START_TAG,i)
            for f in old_features:
                if f not in update:
                    update[f] = -1.0
                else:
                    update[f] -= 1.0
            for f in new_features:
                if f not in update:
                    update[f] = 1.0
                else:
                    update[f] += 1.0
            
    return update
    
def estimate_perceptron(labeled_instances,feat_func,tagger,N_its,all_tags=None):
    """Estimate a structured perceptron by calling sp_update to compute the update
    after each instance
    """

    # List of possible tags
    if all_tags is None:
        all_tags = set()
        for tokens,tags in labeled_instances:
            all_tags.update(tags)


    weights = defaultdict(float, {('NOUN',constants.OFFSET):1e-3})
    weight_history = []    
    avg_weights = defaultdict()
    for it in xrange(N_its):
        w_sum = defaultdict()
        print ("iter",it)
        t = 0.0
        for x_i,y_i in labeled_instances:
            update = sp_update(x_i,y_i,weights,feat_func,tagger,all_tags)
            if len(update)>0:
                for m,n in update.iteritems():
                    weights[m] += n
                    #Update weighted sum
                    if m in w_sum:
                        w_sum[m] += float(t*n)
                    else:
                        w_sum[m] = float(t*n)
                t += 1.0
        #Compute avg weight    
        avg_weights.clear()
        for f in weights:
            avg_weights[f] = weights[f] - ( float(w_sum[f])/t if f in w_sum else 0) 
        weight_history.append(avg_weights.copy())
        weights = defaultdict(float,avg_weights.copy())
    result = { k:v for k, v in weights.items() if v }  #remove pairs with value=0
    return result, weight_history



