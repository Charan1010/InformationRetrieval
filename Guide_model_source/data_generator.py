
# coding: utf-8

# In[11]:

__author__ = "charan_reddy"
import numpy as np
import pandas as pd
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import sys
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer

from gensim.summarization.bm25 import BM25
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import ast
import random
import pickle
from pseudo_feedback import pseudo_relevance_feedback

stop_words = set(stopwords.words('english'))
lemma = nltk.WordNetLemmatizer()


print("loading word 2vec model", sep=' ', end='\n', file=sys.stdout, flush=False)
model = Doc2Vec.load("models/custom_word2vec.model")
print("loading model complete")

print("reading csv files")
df2 = pd.read_csv("csv_files/second_mortgage_csv")
df3 = pd.read_csv("csv_files/ytd_updated.csv")
print("read complete", sep=' ', end='\n', file=sys.stdout, flush=False)


def pre_processing(inputx):
    inputx = inputx.lower()
    words = inputx.split()
    words = [word for word in words if word not in stop_words]
    words = [lemma.lemmatize(i, "v") for i in words]
    outputx = " ".join(words).replace("?", "")
    outputx = outputx.replace('[^\w\s]', '')
    return outputx


print("collecting the input queries from csv file", sep=' ', end='\n', file=sys.stdout, flush=False)
issues = set(df3.RequestDetails.dropna())
issues = [pre_processing(i) for i in issues]
issues = set(issues)
issues = list(issues)
print("query collection complete", sep=' ', end='\n', file=sys.stdout, flush=False)

# The Training data dictionary
d = {"input": [], "g_t": []}

print("loading bm25 object")
with open('bm25_objects/bm252', 'rb') as config_dictionary_file:
    bm252 = pickle.load(config_dictionary_file)
print("loading bm25 object complete")


def data_generator(query, feedback_flag):

    # preprocessing of query started
    query = pre_processing(query)
    # preprocessing of query done

    # generating bm25 scores for guides
    average_idf2 = sum(float(val) for val in bm252.idf.values()) / len(bm252.idf)
    bm25_scores2 = bm252.get_scores(query.split(), average_idf2)
    # bm25 scores generated

    x = np.array(bm25_scores2).argsort(axis=0)[-df2.shape[0]:]

    guide_list = []

    for n, i in enumerate(np.flip(x)):
        guide_list.append(i)

    if feedback_flag == 1:
        query_tokens = word_tokenize(query.lower())
        query_vector = model.infer_vector(query_tokens)
        d["input"].append(query_vector)
        query = pseudo_relevance_feedback(guide_list, query, bm252.idf, 1)
        return data_generator(query, 0)

    query_tokens = word_tokenize(query.lower())
    query_vector = model.infer_vector(query_tokens)
    d["input"].append(query_vector)
    ground_truth = [0]*514
    for i in guide_list[0:1]:
        ground_truth[i] = 1
    d["g_t"].append(ground_truth)
    d["g_t"].append(ground_truth)

    return guide_list


# Data generation loop
for n, issue in enumerate(issues):
    print("----------------------------------------------------------------------------------------------------------")
    print("ITERATION")
    print(n)
    guide_list = data_generator(issue, 1)


print("saving the generated training data")
# saving the generated training data
np.save('training_data/input.npy', d["input"])
np.save("training_data/ground_truth.npy", d["g_t"])
