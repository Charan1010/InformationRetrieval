import pprint
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.summarization.bm25 import BM25
stop_words = set(stopwords.words('english'))
lemma = nltk.WordNetLemmatizer()
from multiprocessing import Pool
import pickle

print("loading all tickets")
df1=pd.read_csv("csv_files/all_tickets.csv")

def pre_processing(inputx):  
    try:
        inputx=inputx.lower()
    except:
        inputx=" "
    words = inputx.split() 
    words = [word for word in words if word not in stop_words] 
    words = [lemma.lemmatize(i,"v") for i in words]
    outputx=" ".join(words).replace("?","")
    outputx=outputx.replace('[^\w\s]','')
    return outputx

print("loading bm25 objects")
with open('bm25_objects/bm25_object1', 'rb') as config_dictionary_file:
    bm251=pickle.load(config_dictionary_file)
        
with open('bm25_objects/bm25_object2', 'rb') as config_dictionary_file:
    bm252=pickle.load(config_dictionary_file)
print("loading done")

def wrapper(query):
    #print("reading generated csv")
    
    print("pre processing query")
    query=pre_processing(query)
    
   
        
    print("bm25 generation1")
    average_idf1 = sum(float(val) for val in bm251.idf.values()) / len(bm251.idf)
    bm25_scores1 = bm251.get_scores(query.split(), average_idf1)

    print("bm25 generation2")
    
    average_idf2 = sum(float(val) for val in bm252.idf.values()) / len(bm252.idf)
    bm25_scores2 = bm252.get_scores(query.split(), average_idf2)

   

    arr=np.array([(0.7*bm25_scores1[i]+bm25_scores2[i])/2 for i in range(0, df1.shape[0])])
    x =arr.argsort(axis=0)[-10:]
    print("getting confidences")
    confidences=[]
    for i in np.fliplr([x])[0]:
        confidences.append(arr[i])
    resolutions=[]
    for i in np.fliplr([x])[0]:
        resolutions.append(df1.Resolution[i])
    score=[]
    print("Using page rank algorithm to swap the retrieved resolutions")
    print("\n\n")
    dict_d={}
    for i in range(len(resolutions)):
        corpus = [resolutions[j].split()  for j in range(len(resolutions))if j!=i]
        bm25=BM25(corpus)
        average_idf = sum(float(val) for val in bm25.idf.values()) / len(bm25.idf)
        bm25_scores = bm25.get_scores(resolutions[i].split(), average_idf)
        score.append(sum(bm25_scores))
        
    score=[(score[i]+confidences[i])/2 for i in range(len(score))]
    
    for i in range(len(resolutions)):
        dict_d[score[i]]=resolutions[i]
  
    final_confidences=sorted(dict_d.keys(),reverse=True)
    final_resolutions=[]
    for i in final_confidences:
        final_resolutions.append(dict_d[i])
        
    return (confidences,final_resolutions)


#pprint.pprint(wrapper("outlook not opening"))

