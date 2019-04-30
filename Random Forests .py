
from random import seed
#Return a randomly selected element from range(start, stop, step). 
from random import randrange
#read CSV file (dataset)
from csv import reader
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


def load_csv(filename):
    dataset=list()
    with open(filename,"r") as file:
        csv_reader=reader(file)
        for row in csv_reader:
            if row=='e,m,p,t,y,!':
                continue
            dataset.append(row)
    return dataset
def stringToint(dataset):

    dbset=list()
    for i in range(len(dataset)):
        intdb=list()
        intdb2=list()
        str,str2,str3,str4=dataset[i]
        temp1=True;
        temp2=True;
        for j in range(maximum(int(str3),int(str4))):
                temp=list()
                if j==int(str3):
                    temp1=False
                if j==int(str4):
                    temp2=False
                if(temp1):
                    if str[j] == 'A':
                        intdb.append('1')
                    if str[j] == 'C':
                        intdb.append('2')
                    if str[j] == 'G':
                        intdb.append('3')
                    if str[j] == 'T':
                        intdb.append('4')
                if (temp2):
                    if str2[j] == 'A':
                        intdb2.append('1')
                    if str2[j] == 'C':
                        intdb2.append('2')
                    if str2[j] == 'G':
                        intdb2.append('3')
                    if str2[j] == 'T':
                        intdb2.append('4')
        temp.append(intdb)
        temp.append(intdb2)
        dbset.append(temp)

    return dbset

def polishData(dataset):
    dbset=list()
    for i in range(len(dataset)):
        temp=list()
        str1,str2=dataset[i]
        temp.append(str1)
        temp.append(str2)
        dbset.append(temp)
    return dbset
def maximum(a,b):

    if a>b:
        return a
    else:
        return b

def merge(dat,dat2):
	for i in range(len(dat)):
		dat[i].append("1")
	for i in range(len(dat2)):
		dat2[i].append("0")

	df=list()
	df.append(dat)
	df.append(dat2)
	dp=list()
	dp.append(df)
	return dp
def str_column_to_float(dataset, column):

    for row in dataset:
        row[column] = float(row[column].strip())
 

def str_column_to_int(dataset, column):

    class_values = [row[column] for row in dataset]

    unique = set(class_values)

    lookup = dict()

    for i, value in enumerate(unique):

        lookup[value] = i

    for row in dataset:
        row[column] = lookup[row[column]]

    return lookup
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def test_split(index, value, dataset):

    left, right = list(), list()

    for row in dataset:

        if row[index] < value:

            left.append(row)
        else:

            right.append(row)

    return left, right
def accuracy_metric(actual, predicted):

    correct = 0

    for i in range(len(actual)):

        if actual[i] == predicted[i]:

            correct += 1

    return correct / float(len(actual)) * 100.0

def evaluate_algorithm(dataset, algorithm, n_folds, *args):

    folds = cross_validation_split(dataset, n_folds)
    scores = list()

    for fold in folds:

        train_set = list(folds)

        train_set.remove(fold)
        train_set = sum(train_set, [])

        test_set = list()

        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None

        predicted = algorithm(train_set, test_set, *args)

        actual = [row[-1] for row in fold]

        accuracy = accuracy_metric(actual, predicted)

        scores.append(accuracy)

    return scores
 
 

def gini_index(groups, class_values):
    gini = 0.0

    for class_value in class_values:

        for group in groups:
            size = len(group)
            if size == 0:
                continue

            proportion = [row[-1] for row in group].count(class_value) / float(size)

            gini += (proportion * (1.0 - proportion))
    return gini
 

def get_split(dataset, n_features):

    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0])-1)
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:

            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups

    return {'index':b_index, 'value':b_value, 'groups':b_groups}


def to_terminal(group):

    outcomes = [row[-1] for row in group]

    return max(set(outcomes), key=outcomes.count)
 

def split(node, max_depth, min_size, n_features, depth):

    left, right = node['groups']
    del(node['groups'])

    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return

    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth+1)

    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)
 
#Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
    #Building the tree involves creating the root node and 
    root = get_split(train, n_features)
    #calling the split() function that then calls itself recursively to build out the whole tree.
    split(root, max_depth, min_size, n_features, 1)
    return root
 
# Make a prediction with a decision tree
def predict(node, row):

    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return(predictions)
 
# Test the random forest algorithm
seed(1)
# load and prepare data
filename = '/home/meet/PycharmProjects/A5/seq_positive_training.csv'
filename2 = '/home/meet/PycharmProjects/A5/seq_negative_training.csv'

dat = load_csv(filename)
dat2=load_csv(filename2)
dat=stringToint(dat)
dat2=stringToint(dat2)
dp=merge(dat,dat2)



df=pd.DataFrame(dp)
print(np.array(df))

filename = "/home/meet/PycharmProjects/A5/kmer_positive_training.csv"
dataset = load_csv(filename)
""""
n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_features = int(sqrt(len(dataset[0])-1))
for n_trees in [1]:
    scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
    print('Trees: %d' % n_trees)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    """




