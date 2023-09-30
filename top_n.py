# Databricks notebook source
#!/usr/bin/env python
# coding: utf-8
import os
from numpy import dot
from numpy.linalg import norm

import gensim
from gensim.models import Word2Vec
from gensim.models import FastText

import pandas as pd
import numpy as np

import gensim
from gensim.models import KeyedVectors
from gensim.models.fasttext import FastText

from matplotlib import pyplot

import string
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# Import local modules
from read_data import read_data
from preprocessing import preprocessing_input
from return_embed import get_mean_vector
import pickle

def cos_sim(a, b):
    """
    Calculate cosine similarity between two vectors a and b.
    """
    return dot(a, b) / (norm(a) * norm(b))

# Function to return top n similar results
def top_n(query, model_name, column_name):
    """
    Return top n similar results for a given query using a specific word embedding model.
    """
    vector_size = 100
    window_size = 3
    df = read_data()

    if model_name == 'Skipgram':
        word2vec_model = Word2Vec.load('https://medicalembeddings.blob.core.windows.net/testcontainer/data/output/skipgram.bin?sp=r&st=2023-09-30T13:41:32Z&se=2023-09-30T21:41:32Z&spr=https&sv=2022-11-02&sr=b&sig=beHxJn7g1KKl2kS8RKhIysNCCIatf6N8dCQQ7Hf5gPc%3D')
        K = pd.read_csv('https://medicalembeddings.blob.core.windows.net/testcontainer/data/output/skipgram_vec.csv?sp=r&st=2023-09-30T13:40:54Z&se=2023-11-10T21:40:54Z&spr=https&sv=2022-11-02&sr=b&sig=AJ1JgZspt16u%2BSxhQtn5Tu6AqixyGcLBsim%2BriAcaRc%3D')
    else:
        word2vec_model = Word2Vec.load('https://medicalembeddings.blob.core.windows.net/testcontainer/data/output/FastText.bin?sp=r&st=2023-09-30T13:40:13Z&se=2023-11-10T21:40:13Z&spr=https&sv=2022-11-02&sr=b&sig=H%2BbCgcmqr2%2FJ9bBCMQTm%2Fd7m5l7uJKCMs6C973RISx8%3D')
        K = pd.read_csv('https://medicalembeddings.blob.core.windows.net/testcontainer/data/output/FastText_vec.csv?sp=r&st=2023-09-30T13:39:17Z&se=2023-11-10T21:39:17Z&spr=https&sv=2022-11-02&sr=b&sig=S4DAvei4Ievlh8Gq8f0ezALnzBI1t%2FwGNqiZ7h0YPdY%3D')

    # Preprocess the input query
    query = preprocessing_input(query)

    # Get the query vector
    query_vector = get_mean_vector(word2vec_model, query)

    # Model Vectors
    # Loading our pretrained vectors of each abstracts

    p = []  # transforming dataframe into required array-like structure as we did in the above step
    for i in range(df.shape[0]):
        p.append(K[str(i)].values)
    x = []
    # Converting cosine similarities of the overall data set with input queries into a LIST
    for i in range(len(p)):
        x.append(cos_sim(query_vector, p[i]))

    # Store list in tmp to retrieve index
    tmp = list(x)

    # Sort list so that the largest elements are on the far right

    res = sorted(range(len(x)), key=lambda sub: x[sub])[-10:]
    sim = [tmp[i] for i in reversed(res)]

    # Get the index of the 10 or n largest element
    L = []
    for i in reversed(res):
        L.append(i)

    df1 = read_data()
    return df1.iloc[L, [1, 2, 5, 6]], sim
