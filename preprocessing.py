# Databricks notebook source
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import string
import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

import gensim
from gensim.models import Word2Vec
from gensim.models import FastText

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# Function to remove all URLs from text
def remove_urls(text):
    """
    Remove URLs from the input text.
    """
    new_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    return new_text

# Make all text lowercase
def text_lowercase(text):
    """
    Convert text to lowercase.
    """
    return text.lower()

# Remove numbers from text
def remove_numbers(text):
    """
    Remove numbers from the input text.
    """
    result = re.sub(r'\d+', '', text)
    return result

# Remove punctuation from text
def remove_punctuation(text):
    """
    Remove punctuation from the input text.
    """
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# Tokenize text
def tokenize(text):
    """
    Tokenize the input text.
    """
    text = word_tokenize(text)
    return text

# Remove stopwords from text
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    """
    Remove stopwords from the input text.
    """
    text = [i for i in text if not i in stop_words]
    return text

# Lemmatize words in text
lemmatizer = WordNetLemmatizer()
def lemmatize(text):
    """
    Lemmatize words in the input text.
    """
    text = [lemmatizer.lemmatize(token) for token in text]
    return text

# Function to preprocess text data
def preprocessing(text):
    """
    Preprocess the input text by applying various text processing steps.
    """
    text = text_lowercase(text)
    text = remove_urls(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    text = ' '.join(text)
    return text

# Function to preprocess an entire DataFrame column
def output_text(df, column_name):
    """
    Preprocess a column in a DataFrame and return the preprocessed text as a list.
    """
    for i in range(df.shape[0]):
        df[column_name][i] = preprocessing(str(df[column_name][i]))
    for i in range(len(df[column_name])):
        df[column_name][i] = df[column_name][i].replace('\n', ' ')
    x = [word_tokenize(word) for word in df[column_name]]
    return x

# Function to preprocess input text
def preprocessing_input(query):
    """
    Preprocess input text for consistency with training data.
    """
    query = preprocessing(query)
    query = query.replace('\n', ' ')
    return query