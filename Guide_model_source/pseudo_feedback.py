from gensim.models.doc2vec import Doc2Vec
from math import sqrt
import numpy as np
import pandas as pd
import operator
model=Doc2Vec.load("models/custom_word2vec.model")
df=pd.read_csv("csv_files/second_mortgage_csv")
def pseudo_relevance_feedback(sorted_guide_list,query,idf,k):
    print("starting pseudo feedback")
    #global feedback_flag
    new_query = query
    relevance_index = {}
    non_relevance_index = {}
    mag_rel = 0
    mag_non_rel = 0
    query_vector = {}
    updated_query = {}
    #consider the top k documents for the feedback loop
    #### making the query vector
    for term in query.split():
        if term in query_vector:
            query_vector[term] +=1
        else:
            query_vector[term] = 1
    for term in idf:
        if not term in query_vector:
            query_vector[term] = 0
    print("query vector generated")
    #print(query_vector)
    ###making the relevant document set vector
    for i in range(0,k):
        guide_id = sorted_guide_list[i]
        guide= df.head_guide_test1[guide_id]
        #print(df.Guide[guide_id])
        #print("\n")
        for term in guide.split():
            if term in relevance_index:
                relevance_index[term] += 1
            else:
                relevance_index[term] = 1

    for term in idf:
        if term in relevance_index:
            relevance_index[term] = relevance_index[term]
        else:
            relevance_index[term] = 0
    ### calculating the magnitude of the relevant document set vector
    for term in relevance_index:
        mag_rel += float(relevance_index[term]**2)
        mag_rel = float(sqrt(mag_rel))
    print("relevant vector generated")
    #print("relevant magnitude" + str(mag_rel))
    ###making the non-relevant document set vector
    for i in range(k+1,len(sorted_guide_list)):
        guide_id = sorted_guide_list[i]
        guide= df.head_guide_test1[guide_id]
        for term in guide.split():
            if term in non_relevance_index:
                non_relevance_index[term] += 1
            else:
                non_relevance_index[term] = 1

    for term in idf:
        if term in non_relevance_index:
            non_relevance_index[term] = non_relevance_index[term]
        else:
            non_relevance_index[term] = 0
    print("non relevant vector generated")
    #print(non_relevance_index)

    ### calculating the magnitude of the relevant document set vector
    for term in non_relevance_index:
        mag_non_rel += float(non_relevance_index[term]**2)
    mag_non_rel = float(sqrt(mag_non_rel))
    #print("non-relevant magnitude" + str(mag_non_rel))
    ###calculating the new query
    for term in idf:
        updated_query[term] = query_vector[term] + (0.5/mag_rel) * relevance_index[term] - (0.15/mag_non_rel) * non_relevance_index[term]

    sorted_updated_query = sorted(updated_query.items(), key=operator.itemgetter(1), reverse=True)
    print("Rocchio algorithm scores generated")

    for i in range(20):
        term,frequency = sorted_updated_query[i]
        #print(term)
        #print(frequency)
        if term not in query:
            for i in query.split():
                try:
                    sim=model.similarity(i,term)
                    #print(sim)
                    if sim>0.45:
                        new_query +=  " "
                        new_query +=  term
                        break
                except Exception as e:
                    pass
    print("ending pseudo feedback")
    return new_query

