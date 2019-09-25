#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Main file to call in functions in intent classifyer.py
#Import Needed things
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer 
import nltk
import re
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

import functions

#loading dataset named "Dataset.csv"
intent, unique_intent, sentences = functions.load_dataset("Dataset.csv")

#Print first 5 sentences
print(sentences[:5])

#download Natural Language tool kit
functions.downloadnltk()

stemmer = LancasterStemmer()

#clean words of puncuation/special characters. Also lemmatiz
cleaned_words = functions.cleaning(sentences)

#2.########################## Encoding ##################
#using tokenizer (a class of keras)
word_tokenizer = functions.create_tokenizer(cleaned_words)
vocab_size = len(word_tokenizer.word_index) + 1
max_length = functions.max_length(cleaned_words)
encoded_doc = functions.encoding_doc(word_tokenizer, cleaned_words)

padded_doc = functions.padding_doc(encoded_doc, max_length)
#print("Shape of padded docs = ",padded_doc.shape)

#changed the filter to keep . and _ as they are present in the strings and are
#necissary.
output_tokenizer = functions.create_tokenizer(unique_intent, filters = '!"#$%&()*+,-/:;<=>?@[\]^`{|}~')
output_tokenizer.word_index

encoded_output = functions.encoding_doc(output_tokenizer, intent)      

encoded_output = functions.np.array(encoded_output).reshape(len(encoded_output), 1)
encoded_output.shape

output_one_hot = functions.one_hot(encoded_output)
output_one_hot.shape

#3. ######################## Train Validation Set ################


#Setting variables  used for training
# Splitting into 80% for training and 20% for validation 
train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot,shuffle = True,test_size = 0.2)
print("Shape of train_X = %s and train_Y = %s" % (train_X.shape, train_Y.shape))
print("Shape of val_X = %s and val_Y = %s" % (val_X.shape, val_Y.shape))

#Defining the Model
#Using Bidirectional GRU 
#GRU is a Gated Recurrent Unit 
#Think its using LSTM and RNNs

model = functions.create_model(vocab_size, max_length)

#Trained Model with adam optimizer
#batch size 16 and epochs 100
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.summary()

filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#Training the Model!!!
#hist = model.fit(train_X, train_Y, epochs = 100, batch_size = 32, validation_data = (val_X, val_Y), callbacks = [checkpoint])
model = load_model("model.h5")

text = "Can you help me?"

def predictions(text):
  lemmatizer = WordNetLemmatizer() 
  print(text)
  clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
  print(clean)
  test_word = word_tokenize(clean)
  print(test_word)
  test_word = [lemmatizer.lemmatize(w.lower()) for w in test_word]
  test_ls = word_tokenizer.texts_to_sequences(test_word)
  
  #Check for unknown words
  if [] in test_ls:
    test_ls = list(filter(None, test_ls))
    
  test_ls = np.array(test_ls).reshape(1, len(test_ls))
 
  x = functions.padding_doc(test_ls, max_length)
  print(x)
  pred = model.predict_classes(x)
  return pred

pred = predictions(text)
print(pred)

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

#predict intent and display the percentage of each
predictions = model.predict(val_X, batch_size=1, verbose=0)

for i in predictions:
        print(i)
     
#predict intent but round to which category it believes it is
rounded_predictions = model.predict_classes(val_X, batch_size=10, verbose=0)

for i in rounded_predictions:
    print(i)
    
classifications = []

for elem in val_Y:
    classifications.append(np.where(elem == 1)[0][0])

cm = confusion_matrix(classifications, rounded_predictions)

print(cm)

plt.imshow(cm, cmap='binary')
plt.show()












