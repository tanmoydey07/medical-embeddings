# Databricks notebook source
import pandas as pd
import numpy as np
import pickle
import gensim
from gensim.models import Word2Vec
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from numpy import dot
from numpy.linalg import norm

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

import gensim
from gensim.models import Word2Vec
from gensim.models import FastText

import pickle
from gensim.models import Word2Vec, FastText

def model_train(x, vector_size, window_size, model):
    """
    Train and save a word embedding model (Word2Vec or FastText).

    Args:
        x (list): List of tokenized sentences for training.
        vector_size (int): Size of the word vectors.
        window_size (int): Maximum distance between the current and predicted word within a sentence.
        model (str): Name of the model to train ('Skipgram' or 'Fasttext').

    Returns:
        gensim.models: Trained Word2Vec or FastText model.
    """
    if model == 'Skipgram':
        skipgram = Word2Vec(x, vector_size=vector_size, window=window_size, min_count=2, sg=1)
        skipgram.save('/dbfs/mnt/data/data/output/skipgram.bin')
        return skipgram
    elif model == 'Fasttext':
        fast_text = FastText(x, vector_size=vector_size, window=window_size, min_count=2, workers=5, min_n=1, max_n=2, sg=1)
        with open('/dbfs/mnt/data/data/output/FastText.bin', 'wb') as f_out:
            pickle.dump(fast_text, f_out)
        f_out.close()
        return fast_text