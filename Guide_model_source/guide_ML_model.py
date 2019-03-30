
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
from keras import optimizers
from keras.models import load_model
import sys
stop_words = set(stopwords.words('english'))
lemma = nltk.WordNetLemmatizer()


print("loading data and vector model files ", sep=' ', end='\n', file=sys.stdout, flush=False)
df = pd.read_csv("csv_files/second_mortgage_csv")
vector_model = Doc2Vec.load("models/d2v_v1.0.model")
print("loading done")

# call this when u have new training data to train the model
"""
print("loading generated training data")
x_train=np.load("training_data/input.npy")
y_train=np.load("training_data/ground_truth.npy")
print("training data loaded", sep=' ', end='\n', file=sys.stdout, flush=False)
"""

# loading pre trained model
model = load_model("models/guide_ml.h5")


# call this when u need to append data to existing training data  files

"""
x_train_append=np.load("training_data/input_append.npy")
y_train_append=np.load("training_data/ground_truth_append.npy")
x_train= np.concatenate((x_train, x_train_append))
y_train= np.concatenate((y_train, y_train_append))

"""

# main model
# Dont call this when loading a pre trained model


def get_model():
    data_dim = 50
    timesteps = 150
    model = Sequential()

    model.add(Dense(128, activation='relu', input_shape=(50,)))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(514, activation='softmax'))
    rmsprop = optimizers.RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=200,
              batch_size=20)


# In[38]:


# training pre trained model on new+old data
#model.fit(x_train, y_train,epochs=500,  batch_size=500,shuffle=True)


# In[70]:


def pre_processing(inputx):
    inputx = inputx.lower()
    words = inputx.split()
    words = [word for word in words if word not in stop_words]
    words = [lemma.lemmatize(i, "v") for i in words]
    outputx = " ".join(words).replace("?", "")
    outputx = outputx.replace('[^\w\s]', '')
    return outputx


def top_guides(query):
    query = pre_processing(query)
    query_tokens = word_tokenize(query.lower())
    query_vector = vector_model.infer_vector(query_tokens)
    print(query_vector)
    list_n = np.fliplr(model.predict(np.array(query_vector).reshape(
        1, -1), batch_size=1).argsort())[0][0:4]
    print(list_n[0:4])
    for i in list_n[0:4]:
        print(df.Heading[i])
