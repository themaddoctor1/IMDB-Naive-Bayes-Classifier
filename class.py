import sys, math, re
import csv

import numpy as np
import matplotlib.pyplot as plt

# Information about the category to classify information into.
categories = [str(int(i/10))+'.'+str(i%10) for i in range(101)]
C_NAME = 'imdb_score'

# Types used for parsing.
f_type = {
    'director_name' : 'str',
    'duration' : 'num',
    'genres' : 'strs',
    'plot_keywords' : 'strs',
    'actor_(.*)_name' : 'regex',
    'content_rating' : 'str',
    'language' : 'str',
    'country' : 'str',
    'gross' : 'num',
    'budget' : 'num',
    'title_year' : 'num',
    'facenumber_in_poster' : 'num',
}

# The list of features that are recognized.
features = [f for f in f_type]

f_filters = {f : (lambda x: [x]) if f_type[f] != 'strs' else (lambda x : x.split('|')) for f in features}
f_index = {}

# Display informative information if needed
if len(sys.argv) < 2 or sys.argv[1] == 'help':
        # Helpful information
        print('usage: python', sys.argv[0], '<csv_file> (<f_name> <f_value>)*')
        print('The following labels are available:')
        for f in features:
            print('',f)
        exit()

# Open the CSV for reading
csvfile = open(sys.argv[1])
csvdict = csv.DictReader(csvfile)

# The measurements that will be collected.
measurements = {}

# Probabilities for each event (will eventually be computed)
probabilities = {f : {} for f in features}

# Provide an initial space for the probability of a given category.
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
    
    # Skip rows that are missing information
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
            # We will have to check multiple columns for values.
            obss = []
            for r in row:
                if re.compile(f).match(r):
                    obss.append(row[r])
            observation[f] = obss 
        else: 
            observation[f] = f_filters[f](row[f])
    
    # Adds an occurence of the given category.
    category = row[C_NAME]
    probabilities[C_NAME][category] += 1.0
    
    # Record the measurement(s).
    for f in features:
        for obs in observation[f]:
            add_measurement(measurements, f, obs, category)
    
    #print()
    row_num += 1

print('Found', row_num, 'usable movies')

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
            #print(f, '~ N(', mean, ',', stddev, '|', 'C', '=', c, ')')
            if stddev == 0:
                return float('inf') if mean == float(m) else 0
            else:
                # First, handle z-value
                z = (float(m) - mean) / stddev

                # Then, compute density value
                p = math.exp(-0.5*z**2) / math.sqrt(2*math.pi * stddev**2)

                # Finally, return the density value.
                return p

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
        for f, m in labels:
            # Update the probability
            pr = Pr(f, m, c)
            p *= pr

        if p > C_prob:
            C_max = c
            C_prob = p

    return C_max

if __name__ == '__main__':
     
    # Get the labels to handle.
    my_labels = []
    for i in range(2, len(sys.argv), 2):
        f = sys.argv[i]
        m = sys.argv[i+1]
        my_labels.append((f, m))

    my_preds = []
    
    p_net = 0

    for c in categories:
        # Compute the initial probability
        p = P(c)
        
        for f, m in my_labels:
            # Compute the next part of the probability
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


