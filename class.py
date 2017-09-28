import sys, math, re
import csv

import numpy as np
import matplotlib.pyplot as plt

csvfile = open('./class/movie_metadata.csv')
csvdict = csv.DictReader(csvfile)

categories = [str(int(i/10))+'.'+str(i%10) for i in range(101)]
C_INDEX = -1
C_NAME = 'imdb_score'

features = ['director_name', 'duration', 'actor_1_name', 'actor_2_name', 'genres', 'plot_keywords', 'actor_(.*)_name']
f_type = {
    'director_name' : 'str',
    'duration' : 'num',
    'actor_1_name' : 'str',
    'actor_2_name' : 'str',
    'genres' : 'strs',
    'plot_keywords' : 'strs',
    'actor_(.*)_name' : 'regex',
}
f_filters = {f : (lambda x: [x]) if f_type[f] != 'strs' else (lambda x : x.split('|')) for f in features}
f_index = {}

measurements = {}
probabilities = {f : {} for f in features}
probabilities[C_NAME] = {c : 0 for c in categories}

def add_measurement(measurements, feature, value, category):
    """Adds a measurement to the given measurement dictionary.
    feature - The feature being measured.
    value - The value to record.
    category - The category under which the measurement was made.
    """

    #print('Attempting to add measurement:')
    #print('( feat:', feature, ', val:', value, ', cat:', category, ')')

    # If the feature has not yet been witnessed, provide a storage space.
    if feature not in measurements:
        measurements[feature] = {c : [] for c in categories}
    
    # Store the measurement.
    measurements[feature][category].append(float(value) if f_type[feature] == 'num' else value)

# We will read measurements from the header file.
cols = None
row_num = 0
for row in csvdict:
    
    if any(f_type[f] != 'regex' and row[f] == '' for f in features):
        continue
    
    """
    print('Row', row_num, '\n============')
    for f in features:
        print(f, ':', row[f])
    print('category :', row[C_NAME])
    print()"""

    # Make an observation from the CSV.
    observation = {}
    for f in features:
        if f_type[f] == 'regex':
            obss = []
            for r in row:
                if re.compile(f).match(r):
                    obss.append(row[r])
            observation[f] = obss 
        else: 
            observation[f] = f_filters[f](row[f])

    category = row[C_NAME]
    probabilities[C_NAME][category] += 1.0
    
    # Record the measurement.
    for f in features:
        for obs in observation[f]:
            add_measurement(measurements, f, obs, category)
    
    #print()
    row_num += 1

# Now, we condense the probabilities.
for f in features:
    if f_type[f] == 'str' or f_type[f] == 'strs' or f_type[f] == 'regex':
        #print(f, 'is discrete')
        # The dataset is discrete, so develop such a set.
        for c in categories:
            # Add to the count of each measurement.
            for m in measurements[f][c]:
                if c not in probabilities[f]:
                    probabilities[f][c] = {m : 1.0}
                elif m not in probabilities[f][c]:
                    probabilities[f][c][m] = 1.0
                else:
                    probabilities[f][c][m] += 1.0

            # Now, divide by the number of measurements.
            if c in probabilities[f]:
                for m in probabilities[f][c]:
                    probabilities[f][c][m] /= len(measurements[f][c])
                    #print('P(', f, '=', m, '| C =', c, ') =', probabilities[f][c][m])

            # So, probabilities[f][c][m] gives us P(f=m | c)
    
    elif f_type[f] == 'num':
        #print(f, 'is continuous')
        # The dataset is continuous, so we will develop a normal distribution.
        for c in categories:
            # Compute the mean + stddev
            if len(measurements[f][c]) > 0:
                # Build a normal distribution
                mean = np.mean(measurements[f][c])
                stddev = np.std(measurements[f][c])
                probabilities[f][c] = (mean, stddev)
                #print('P(', f, '= f | C =', c, ') ~ N(', mean, ',', stddev, ')')

# Finally, compute the probabilities of each category.
for c in probabilities[C_NAME]:
    #print('count of', c, 'is', probabilities[C_NAME][c])
    probabilities[C_NAME][c] /= row_num
    #print('P( C =', c, ') =', probabilities[C_NAME][c])

def P(c):
    return probabilities[C_NAME][c]
    
def Pr(f, m, c):
    """Computes P(f = m | c)
    """
    #print('Find P(', f, '=', m, '| C =', c, ')')

    if f_type[f] == 'str' or f_type[f] == 'strs' or f_type[f] == 'regex':
        #print(f, 'is a discrete variable')
        if c not in probabilities[f] or m not in probabilities[f][c]:
            return 0
        else:
            return probabilities[f][c][m]
    else:
        #print(f, 'is a continuous variable')
        if c not in probabilities[f]:
            #print(f, '= 0')
            return 0
        else:
            mean, stddev = probabilities[f][c]
            #print(f, '~ N(', mean, ',', stddev, ')')
            return math.exp(-((float(m) - mean) / stddev)**2) / math.sqrt(2*math.pi)

def NaiveBayes(labels):
    """Performs Naive-Bayes classification. Given a set of labels
    and their values, compute the most likely category to be observed.
    If the function returns None, then none of the categories could
    be given any confidence of being chosen.
    """
    C_max = None
    C_prob = 0
    for c in categories:
        p = P(c)
        # Now, multiply out all of the likelihoods
        for f in labels:
            # Get the observation
            m = labels[f]

            # Update the probability
            pr = Pr(f, m, c)
            p *= pr

        if p > C_prob:
            C_max = c
            C_prob = p

    return C_max

if __name__ == '__main__':
    
    if len(sys.argv) > 1 and sys.argv[1] == 'help':
        print('To run, execute \'python', sys.argv[0], '(<f_name> <f_value>)*\'')
        print('The following labels are available:')
        for f in features:
            print(f)
        exit()

    my_labels = {}
    for i in range(1, len(sys.argv), 2):
        f = sys.argv[i]
        m = sys.argv[i+1]
        my_labels[f] = m

    my_preds = []
    
    p_net = 0

    for c in categories:
        p = P(c)
        
        for f in my_labels:
            m = my_labels[f]

            p *= Pr(f, m, c)
        
        p_net += p
        my_preds.append(p)

    if p_net == 0:
        print('Did not find any instances that match')
        exit()

    for i in range(len(my_preds)):
        my_preds[i] /= p_net

    c = NaiveBayes(my_labels)
    
    # Display a bar chart
    bars = plt.bar(np.arange(len(categories)), my_preds, align='center', color='k')
    bars[categories.index(c)].set_color('green')

    plt.xticks(np.arange(101), [(str(i/10) if i%10 == 0 else '') for i in range(101)])
    plt.xlabel('IMDB Rating')
    plt.ylabel('Prediction Probability')
    plt.title('Optimum: P(C = ' + str(c) + ' | F) = ' + str(my_preds[categories.index(c)]))

    plt.show()


