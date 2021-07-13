# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 16:54:48 2021

@author: Jelson
"""

import os
import platform
import nltk
import random
import pickle

#nltk.download('stopwords')

from re import split
from nltk.corpus import stopwords

#Use this function of output file not present
def combine_files():
    if(platform.system() == 'Windows'):
        os.system("type .\data\*.txt > .\data\output.txt")
    else:
        os.system("cat ./data/*.txt > ./data/output.txt")

#combine_files()

documents = []
read_file = "./data/output.txt"

def create_docs():
    with open(read_file,'r', encoding='utf-8', errors='ignore') as r:
        for line in r:
            splitted = line.strip().split('\t')
            msg = ('').join(splitted[:-1])
            is_class = splitted[-1]
            #documents.extend([ dict(doc=msg.lower(),category=is_class)])
            if(is_class == "1" or is_class == "0"):
                documents.extend([ dict(doc=msg.lower(),category=is_class)])
    return documents

print("Creating document obj...")        
documents = create_docs()

def create_words(document):
    for n in range(len(document)):
        document[n]['words'] = split('\W+', document[n]['doc'])
    return document

print("Creating words...")    
documents = create_words(documents)

def remove_empty_words(document):
    for n in range(len(document)):
        if('' in document[n]['words']):
            document[n]['words'].remove('')
    return document

print("Removing empty words...")
documents = remove_empty_words(documents)

print("Getting frequency count of all words...")
all_words = nltk.FreqDist(w.lower() for d in documents for w in d['words'] if w not in stopwords.words() and not w.isdigit())
    
word_features = all_words.most_common(2000)

def document_features(document):
    if not document.get('words'):
        document['words']=split('\W+', document['doc'])
    document_words=set(document['words'])
    print(document_words)
    features={}
    for word_t in word_features:
        word = word_t[0]
        features["contains('{0}')".format(word)]=(word in document_words)
    return features

random.shuffle(documents)

print("Creating feature sets...")
featuresets=[(document_features(d),d['category']) for d in documents]

print("Creating training and testing sets...")
train_set,test_set = featuresets[:2500],featuresets[2500:]

if(os.path.exists('./pickle') == False):
    print("Running Naive Bayes Classifier...")
    nbc = nltk.NaiveBayesClassifier.train(train_set)
    
    print("Running Decision Tree Classifier...")
    dtc = nltk.DecisionTreeClassifier.train(train_set)
    
    print("Running Maxent Classifier...")
    mc = nltk.MaxentClassifier.train(train_set)
    
    print("Path doesnt exist...")
    print("Creating pickle path...")
    os.mkdir('.\pickle')
    pickle.dump(nbc, open('./pickle/nbc.pkl','wb'))
    pickle.dump(dtc, open('./pickle/dtc.pkl','wb'))
    pickle.dump(mc, open('./pickle/mc.pkl','wb'))
    
else:
    print("Pickle files already exists...")
    print("Loading pickle files...")
    nbc = pickle.load(open('./pickle/nbc.pkl', 'rb'))
    dtc = pickle.load(open('./pickle/dtc.pkl', 'rb'))
    mc = pickle.load(open('./pickle/mc.pkl', 'rb'))
    
print("Accuracy for Naive Bayes Classifier: ",nltk.classify.accuracy(nbc, test_set)*100)
print("Accuracy for Decision Tree Classifier: ",nltk.classify.accuracy(dtc, test_set)*100)
print("Accuracy for Maximum Entropy Classifier: ",nltk.classify.accuracy(mc, test_set)*100)

nbc.show_most_informative_features(25)


while 1:
    statement = input("Enter the sentence: ")
    ob = {'doc': statement}
    
    print ("Naive Bayes says: ",nbc.classify(document_features(ob)))
    print ("Decision Tree says: ",dtc.classify(document_features(ob)))
    print ("Maximum Entropy says: ",mc.classify(document_features(ob)))
