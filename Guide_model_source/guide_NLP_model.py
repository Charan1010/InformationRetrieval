
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
from pseudo_feedback import pseudo_relevance_feedback
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import XMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
from pdfminer.layout import LAParams
from pdfminer.converter import PDFPageAggregator
import pdfminer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from gensim.summarization.bm25 import BM25
from PyPDF2 import PdfFileWriter, PdfFileReader
from PyPDF2.generic import (
    DictionaryObject,
    NumberObject,
    FloatObject,
    NameObject,
    TextStringObject,
    ArrayObject
)
import ast
import random
import subprocess
import cProfile, pstats, io
import pickle
import re
import pprint
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemma=nltk.WordNetLemmatizer()
input_file="pdf_files/HelpDesk Notebook.pdf"
output_file="output/final_output.pdf"


print("reading both csv files")
df1=pd.read_csv("csv_files/first_mortgage_csv")
df2=pd.read_csv("csv_files/second_mortgage_csv")
print("reading both csv complete")


print("reading entire pdf file")
pdfInput = PdfFileReader(open(input_file, "rb"))

Total_pages=1306

print("creating a read object for every page")
pdf=[]
for i in range(Total_pages):
    pdf.append(pdfInput.getPage(i))
print("read objects created")


with open('bm25_objects/bm25', 'rb') as config_dictionary_file:
    bm25=pickle.load(config_dictionary_file)
with open('bm25_objects/bm252', 'rb') as config_dictionary_file:
    bm252=pickle.load(config_dictionary_file)


def pre_processing(inputx):
    inputx = inputx.lower()
    words = inputx.split()
    words = [word for word in words if word not in stop_words]
    words = [lemma.lemmatize(i,"v") for i in words]
    outputx = " ".join(words).replace("?", "")
    outputx = outputx.replace('[^\w\s]', '')
    return outputx



def wrapper(query,feedback_flag):
    
    print("preprocessing of query started")
    query = pre_processing(query)
    print("preprocessing of query done")
                 
    print("calculating bm25 scores for query vs guides")
    average_idf2 = sum(float(val) for val in bm252.idf.values()) / len(bm252.idf)
    
    bm25_scores2 = bm252.get_scores(query.split(), average_idf2)
    print("calculation done")
    
    #y=np.array([[[bm25_scores2[i]]] for i in range(0, df2.shape[0])])
    x = np.array(bm25_scores2).argsort(axis=0)[-df2.shape[0]:]
    Guides = []
    guide_list=[]

    print("ranking guides started")
    for i in np.flip(x):
        guide_list.append(i)
        Guides.append(df2.Heading[i])

    print("ranking guides done")
    print(Guides[0:4])
    
    if feedback_flag==1:
        print("query extension with pseudo relevance feedback started")
        query=pseudo_relevance_feedback(guide_list,query,bm252.idf,1)
        print("query extension done")
        return wrapper(query,0)


    print("calculating bm25 scores for query vs steps")
    average_idf = sum(float(val) for val in bm25.idf.values()) / len(bm25.idf)
    bm25_scores = bm25.get_scores(query.split(), average_idf)
    print("calculation done")

    x = np.array(bm25_scores).argsort(axis=0)[-4:]
    final_list=[]
    print("sorting steps acc to guide priority")
    for j in range(0,df2.shape[0]):
        for i in np.flip(x):
            if Guides.index(df1.Heading[i])==j:
                final_list.append(i)
    print("sorting steps done")
    print(final_list)

    print("sorting steps that belong to same guide")
    for i in range(0,3):
        for j in range(0,3-i):
            if df1.Heading[final_list[j]]==df1.Heading[final_list[j+1]] and final_list[j]>final_list[j+1]:
                final_list[j],final_list[j+1]=final_list[j+1],final_list[j]
    print("sort done")

    confidences=[]
    for i in final_list:
        confidences.append(bm25_scores[i])

    Q_loc=df1.iloc[final_list,[3,1,2,0]]
    
    print("highlight functions initiated")

    def createHighlight(x1, y1, x2, y2, meta, color=[1, 1, 0]):
            print("inside create highlight")
            newHighlight = DictionaryObject()

            newHighlight.update({
                NameObject("/F"): NumberObject(4),
                NameObject("/Type"): NameObject("/Annot"),
                NameObject("/Subtype"): NameObject("/Highlight"),

                NameObject("/T"): TextStringObject(meta["author"]),
                NameObject("/Contents"): TextStringObject(meta["contents"]),

                NameObject("/C"): ArrayObject([FloatObject(c) for c in color]),
                NameObject("/Rect"): ArrayObject([
                    FloatObject(x1),
                    FloatObject(y1),
                    FloatObject(x2),
                    FloatObject(y2)
                ]),
                NameObject("/QuadPoints"): ArrayObject([
                    FloatObject(x1),
                    FloatObject(y2),
                    FloatObject(x2),
                    FloatObject(y2),
                    FloatObject(x1),
                    FloatObject(y1),
                    FloatObject(x2),
                    FloatObject(y1)
                ]),
            })

            return newHighlight

    def addHighlightToPage(highlight, page, output):
            highlight_ref = output._addObject(highlight)
            print(highlight_ref)

            if "/Annots" in page:
                print("Annots in page")
                page[NameObject("/Annots")].append(highlight_ref)
            else:
                print("Annots not in page")
                page[NameObject("/Annots")] = ArrayObject([highlight_ref])
    def highlighter(pdf_file, Q_loc):
        
            print("pdf file being read")
            #pdfInput = PdfFileReader(open(pdf_file, "rb"))
            print("pdf writer outght to be created")
            pdfOutput = PdfFileWriter()

          
            #iteration = 0
            x = 0
            # prev_pages=[]
            for index,i in Q_loc.iterrows():
                #pdfInput = PdfFileReader(open(pdf_file, "rb"))
                #pdfOutput = PdfFileWriter()
                
                pagenum=i["p_num"]
                
                page1 = pdf[pagenum]
                i["Coordinates"]=ast.literal_eval(i["Coordinates"])

                highlight = createHighlight(i["Coordinates"][0], i["Coordinates"][1], 838, i["Coordinates"][3], {
                    "author": "Suprath Tech",
                    "contents": "Please go through the full guide for complete understanding"})
                print("calling addhighlight")
                addHighlightToPage(highlight, page1, pdfOutput)

     
    
    # In[189]:
    print("highlighting the relevant questions based on page numbers and co-ordinates captured")

    highlighter(input_file, Q_loc)

    print("highlight done")

    print("creating new output pdfs and page number key-value pairs")
    pdfOutput = PdfFileWriter()
    guide_names=set(Q_loc["Heading"])
    p_m={}
    k=0
    for i in guide_names:
        slices=df2[df2.Heading==i]["Page_num"]
        slices=str(slices.values[0])
        slices=re.findall(r'\d+',slices)
        slices=list(map(int, slices))
        print(slices)
        if len(slices)>1:
            #print(slices)
            for page_num in range(slices[0],(slices[1]+1)):
                p_m[page_num]=k
                k=k+1
                pdfOutput.addPage(pdf[page_num])
        else:
            p_m[slices[0]]=k
            k=k+1
            pdfOutput.addPage(pdf[slices[0]])
    
    with open(output_file, 'wb') as f:
        pdfOutput.write(f)

    print("creation done")

    results = [output_file+"#page="+str(p_m[row.p_num]+1)+"&zoom=120,72,"+str(792-ast.literal_eval(row.Coordinates)[3]) for index,row in Q_loc.iterrows()]
    print(p_m)
    return results,list(Q_loc.p_num),confidences


# In[172]:


#pprint.pprint(wrapper("how to create a shared mailbox",1))


# In[83]:


